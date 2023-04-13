import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from typing import List, Optional, Any, Union, Tuple, Dict

from ..utils import _init_params


class MHSABlock(nn.Module):
    """
    Implement the transformer module in ViT.
    Args:
        - dim (int): the dimension of the input tensor (batch_size, num_patches, dim).
        - num_heads (int): the number of attention heads.
        - dropout (float): the dropout rate.
        - head_dim (int; optional): the dimension of each atttention head. Will be set to (dim // num_heads) if not specified.
        - mlp_dim (int; optional): the dimension of the hidden layer in the mlp. Will be set to (dim * 2) if not specified.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.5,
        mlp_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        mlp_dim = mlp_dim if mlp_dim is not None else dim * 2

        self.ln_1 = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        _init_params(self)

    def forward(self, input: Tensor) -> Tensor:
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class MHSA(nn.Module):
    def __init__(
        self,
        sources: List[str],
        in_channels: int = 512,
        feat_size: Union[int, Tuple[int, int]] = (1, 7, 7),
        patch_size: Union[int, Tuple[int, int]] = (1, 1, 1),
        num_heads: int = 8,
        depth: int = 2,
        dropout: float = 0.5,
        mask_ratio: float = 0.0,
        fusion_steps: int = 1,
        mlp_dim: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        sources.sort()
        self.sources = sources

        assert fusion_steps in [1, 2]
        if len(self.sources) in [1, 2]:
            self.fusion_steps = 1
        else:
            assert len(self.sources) == 4
            self.fusion_steps = fusion_steps

        self.feat_size = feat_size if isinstance(feat_size, tuple) else (feat_size, feat_size, feat_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, patch_size)
        assert isinstance(self.feat_size, tuple) and len(self.feat_size) == 3
        assert isinstance(self.patch_size, tuple) and len(self.patch_size) == 3
        assert self.feat_size[0] % self.patch_size[0] == 0
        assert self.feat_size[1] % self.patch_size[1] == 0
        assert self.feat_size[2] % self.patch_size[2] == 0
        self.num_pt = self.feat_size[0] // self.patch_size[0]
        self.num_ph, self.num_pw = self.feat_size[1] // self.patch_size[1], self.feat_size[2] // self.patch_size[2]

        patch_dim = int(self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * in_channels)

        no_pos_embeddings = self.num_pt == self.num_ph == self.num_pw == len(self.sources) == 1
        pos_embeddings = None if no_pos_embeddings else torch.randn(len(self.sources), self.num_pt * self.num_ph * self.num_pw, patch_dim)
        for idx in range(len(self.sources)):
            setattr(
                self,
                f"{self.sources[idx]}_embeddings",
                None if no_pos_embeddings else nn.Parameter(pos_embeddings[idx: (idx + 1), :, :])
            )

        self.mask_ratio = mask_ratio

        if self.fusion_steps == 1:
            self.mhsa = nn.Sequential(*[MHSABlock(dim=patch_dim, num_heads=num_heads, dropout=dropout, mlp_dim=mlp_dim) for _ in range(depth)])
        else:
            self.mhsa_top = nn.Sequential(*[MHSABlock(dim=patch_dim, num_heads=num_heads, dropout=dropout, mlp_dim=mlp_dim) for _ in range(depth)])
            self.mhsa_front = nn.Sequential(*[MHSABlock(dim=patch_dim, num_heads=num_heads, dropout=dropout, mlp_dim=mlp_dim) for _ in range(depth)])
            self.mhsa_out = nn.Sequential(*[MHSABlock(dim=patch_dim, num_heads=num_heads, dropout=dropout, mlp_dim=mlp_dim) for _ in range(depth)])

        self.out_dim = patch_dim
        _init_params(self)

    def _mask_patches(self, x: Tensor) -> Tensor:
        assert self.training
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        x_sources = list(x.keys())
        x_sources.sort()
        assert set(x_sources).issubset(self.sources)
        for source in x_sources:
            if x[source].shape[-3:] != self.feat_size:
                x[source] = F.interpolate(x[source], size=self.feat_size, mode="trilinear")

            x[source] = rearrange(
                tensor=x[source],
                pattern="bs c (num_pt pt) (num_ph ph) (num_pw pw) -> bs (num_pt num_ph num_pw) (pt ph pw c)",
                pt=self.patch_size[0],
                ph=self.patch_size[1],
                pw=self.patch_size[2]
            )

            pos_embeddings = getattr(self, f"{source}_embeddings")
            x[source] = x[source] + pos_embeddings if pos_embeddings is not None else x[source]

        if self.fusion_steps == 1:
            x = torch.cat([x[source] for source in x_sources], dim=1)
            if self.mask_ratio > 0 and self.training:
                x = self._mask_patches(x)

            out = self.mhsa(x)
            out = torch.sum(out, dim=1, keepdim=False)

        else:
            assert len(self.sources) == 4
            if "top_IR" in x_sources and "top_depth" in x_sources:
                x_top = torch.cat([x["top_IR"], x["top_depth"]], dim=1)
            elif "top_IR" in x_sources and "top_depth" not in x_sources:
                x_top = x["top_IR"]
            elif "top_IR" not in x_sources and "top_depth" in x_sources:
                x_top = x["top_depth"]
            else:
                x_top = None

            if x_top is not None:
                if self.mask_ratio > 0 and self.training:
                    x = self._mask_patches(x)
                x_top = self.mhsa_top(x_top)

            if "front_IR" in x_sources and "front_depth" in x_sources:
                x_front = torch.cat([x["front_IR"], x["front_depth"]], dim=1)
            elif "front_IR" in x_sources and "front_depth" not in x_sources:
                x_front = x["front_IR"]
            elif "front_IR" not in x_sources and "front_depth" in x_sources:
                x_front = x["front_depth"]
            else:
                x_front = None

            if x_front is not None:
                if self.mask_ratio > 0 and self.training:
                    x = self._mask_patches(x)
                x_front = self.mhsa_front(x_front)

            if x_top is not None and x_front is not None:
                out = torch.cat([x_top, x_front], dim=1)
            elif x_top is None and x_front is not None:
                out = x_front
            elif x_top is not None and x_front is None:
                out = x_top
            else:
                out = None

            if out is not None:
                out = self.mhsa_out(out)
                out = torch.sum(out, dim=1, keepdim=False)

        return out
