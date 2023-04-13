import torch
from torch import nn, Tensor
from typing import Optional, Any


class FocalLoss(nn.Module):
    """
    Implement the focal loss proposed in Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002).
    """
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        gamma: float = 2,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.gamma = gamma

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        ce_loss = self.ce_loss_fn(y_pred, y_true)

        prob = torch.softmax(y_pred, dim=1)
        prob = torch.gather(prob, dim=1, index=y_true.view(-1, 1)).view(-1)

        focal_loss = ((1 - prob) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)
