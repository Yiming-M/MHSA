import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .utils import _init_params, _concat_all_gather


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 256,
        out_dim: int = 128,
        normalize: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        self.normalize = normalize

        _init_params(self)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = F.normalize(x, p=2, dim=1) if self.normalize else x
        return x


class MemoryBank(nn.Module):
    def __init__(self, num_classes: int, dim: int) -> None:
        """
        Create a classifier based on previously encoded features (memory).

        Args:
            memory (Dict[int, Tensor]): the memory with keys being the classes.
        """
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        for i in range(self.num_classes):
            self.register_buffer(f"cls_{i}", torch.zeros(1, self.dim))
            self.register_buffer(f"cls_{i}_ctr", torch.tensor(0).int())

    @torch.no_grad()
    def update_memory(self, cls: int, feat: Tensor) -> None:
        bs, feat_dim = feat.shape
        assert feat_dim == self.dim

        setattr(self, f"cls_{cls}", getattr(self, f"cls_{cls}") + torch.sum(feat, dim=0, keepdim=True))
        setattr(self, f"cls_{cls}_ctr", getattr(self, f"cls_{cls}_ctr") + bs)

    @torch.no_grad()
    def forward(self, feats: Tensor) -> Tensor:
        feats = F.normalize(feats, p=2, dim=1)
        bs, dim = feats.shape
        assert dim == self.dim

        scores = torch.empty(size=(bs, self.num_classes), device=feats.device)
        for cls in range(self.num_classes):
            memory_cls = getattr(self, f"cls_{cls}") / getattr(self, f"cls_{cls}_ctr")
            sims = torch.matmul(feats, memory_cls.T)  # [-1, 1]
            sims = (sims + 1) / 2  # [0, 1]
            scores[:, cls] = torch.squeeze(sims, dim=1)

        return scores


class MemoryBankDDP(nn.Module):
    def __init__(self, num_classes: int, dim: int) -> None:
        """
        Create a classifier based on previously encoded features (memory).

        Args:
            memory (Dict[int, Tensor]): the memory with keys being the classes.
        """
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        for i in range(self.num_classes):
            self.register_buffer(f"cls_{i}", torch.zeros(1, self.dim))
            self.register_buffer(f"cls_{i}_ctr", torch.tensor(0).int())

    @torch.no_grad()
    def update_memory(self, cls: int, feat: Tensor) -> None:
        feat = _concat_all_gather(feat)
        bs, feat_dim = feat.shape
        assert feat_dim == self.dim

        setattr(self, f"cls_{cls}", getattr(self, f"cls_{cls}") + torch.sum(feat, dim=0, keepdim=True))
        setattr(self, f"cls_{cls}_ctr", getattr(self, f"cls_{cls}_ctr") + bs)

    @torch.no_grad()
    def forward(self, feats: Tensor) -> Tensor:
        feats = F.normalize(feats, p=2, dim=1)
        bs, dim = feats.shape
        assert dim == self.dim

        scores = torch.empty(size=(bs, self.num_classes), device=feats.device)
        for cls in range(self.num_classes):
            memory_cls = getattr(self, f"cls_{cls}") / getattr(self, f"cls_{cls}_ctr")
            sims = torch.matmul(feats, memory_cls.T)  # [-1, 1]
            sims = (sims + 1) / 2  # [0, 1]
            scores[:, cls] = torch.squeeze(sims, dim=1)

        return scores
