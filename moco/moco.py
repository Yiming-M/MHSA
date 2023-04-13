# Original code is from https://github.com/facebookresearch/moco/blob/main/moco/builder.py.
# Modified to enable supervised contrastive learning.

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from copy import deepcopy
from typing import Tuple


class SuMoCo(nn.Module):
    def __init__(
        self,
        base_encoder: nn.Module,
        mlp: nn.Module,
        dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
    ) -> None:
        """
        Supervised contrastive learning with MoCoV2.
        This module will return encoded features and their corresponding labels.

        Args:
            base_encoder (nn.Module): the backbone model, which returns normalised features.
            mlp (nn.Module): the projection head, which comprises FC -> ReLU -> FC.
            dim (int, optional): feature dimension. Defaults to 128.
            K (int, optional): queue size. Defaults to 65536.
            m (float, optional): moco momentum of updating key encoder. Defaults to 0.999.
        """
        super().__init__()
        self.K = K
        self.m = m

        self.encoder_q = base_encoder
        self.encoder_k = deepcopy(base_encoder)
        self.mlp_q = mlp
        self.mlp_k = deepcopy(mlp)
        self.__init_params_k__()

        # create the queues.
        self.register_buffer("queue_feats", torch.randn(dim, K))
        self.queue_feats = F.normalize(self.queue_feats, p=2, dim=0)
        self.register_buffer("queue_labels", torch.zeros(K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def __init_params_k__(self) -> None:
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialisation
            param_k.requires_grad = False  # not updated by gradient

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialisation
            param_k.requires_grad = False  # not updated by gradient

    @torch.no_grad()
    def __momentum_update_params_k__(self) -> None:
        """
        Momentum update of the parameters of the key encoder and the mlp.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def __dequeue_and_enqueue__(self, feats: Tensor, labels: Tensor) -> None:
        assert feats.shape[0] == labels.shape[0]
        batch_size = feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the feats and labels at ptr (dequeue and enqueue)
        self.queue_feats[:, ptr: ptr + batch_size] = feats.T
        self.queue_labels[ptr: ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer.

        self.queue_ptr[0] = ptr

    def forward_train(self, x_q: Tensor, x_k: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # compute query features
        hidden_feats = self.encoder_q(x_q)  # queries: NxC
        q = F.normalize(self.mlp_q(hidden_feats), p=2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self.__momentum_update_params_k__()  # update the key encoder

            k = self.encoder_k(x_k)  # keys: NxC
            k = F.normalize(self.mlp_k(k), p=2, dim=1)

        queue_feats = self.queue_feats.clone().t()
        queue_labels = self.queue_labels.clone()

        # dequeue and enqueue
        self.__dequeue_and_enqueue__(k, y)

        return q, k, queue_feats, queue_labels, hidden_feats.detach()

    @torch.no_grad()
    def foward_test(self, x_q: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.encoder_q(x_q)

    def forward(self, x_q: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        if self.training:
            return self.forward_train(x_q, **kwargs)
        else:
            return self.foward_test(x_q)
