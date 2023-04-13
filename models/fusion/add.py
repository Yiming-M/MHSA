import torch
from torch import nn, Tensor

from typing import List, Any, Dict


class Add(nn.Module):
    def __init__(
        self,
        sources: List[str],
        in_channels: int,
        mask_ratio: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__()
        sources.sort()
        self.sources = sources
        self.out_dim = in_channels
        self.mask_ratio = mask_ratio
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()

    def _mask_mods(self, x: Tensor) -> Tensor:
        assert self.training
        bs, m, c, t, h, w = x.shape
        ids_keep = int(m * (1 - self.mask_ratio))
        noise = torch.rand(bs, m, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # keep the first subset
        ids_keep = ids_shuffle[:, :ids_keep]
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, c, t, h, w))
        return x

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        x_sources = list(x.keys())
        x_sources.sort()
        assert set(x_sources).issubset(self.sources)

        x = torch.stack([x[source] for source in x_sources], dim=1)
        if self.mask_ratio > 0 and self.training:
            x = self._mask_mods(x)

        x = torch.sum(x, dim=1, keepdim=False)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return x
