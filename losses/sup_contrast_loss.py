# modified from https://github.com/zhangyifei01/MoCo-v2-SupContrast/blob/main/sup_losses.py

import torch
from torch import nn, Tensor

ESP = 1e-6


class SupConLoss(nn.Module):
    def __init__(self, T: float = 0.07):
        """
        Supervised contrastive learning loss for classification.
        This loss function is used to handle the outputs from SupMoCo.

        Args:
            T (float, optional): softmax temperature. Defaults to 0.07.
        """
        super().__init__()
        self.T = T

    def forward(self, q: Tensor, k: Tensor, y: Tensor, queue_feats: Tensor, queue_labels: Tensor) -> Tensor:
        batch_size, device = q.shape[0], q.device

        feats, labels = torch.cat([q, k, queue_feats], dim=0), torch.cat([y, y, queue_labels], dim=0)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(feats[:batch_size], feats.T), self.T)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + ESP)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos.mean()

        return loss


class DetConLoss(nn.Module):
    def __init__(self, T: float = 0.07):
        """
        Supervised contrastive learning loss for detection.
        Only normal data are pushed to be close. Normal and anomalous data should be far away from each other.
        This loss function is used to handle the outputs from SuMoCo.

        Args:
            T (float, optional): softmax temperature. Defaults to 0.07.
        """
        super().__init__()
        self.T = T

    def forward(self, q: Tensor, k: Tensor, y: Tensor, queue_feats: Tensor, queue_labels: Tensor) -> Tensor:
        mask_n = y == 0
        q_n, q_a = q[mask_n], q[~mask_n]
        k_n, k_a = k[mask_n], k[~mask_n]

        mask_n = queue_labels == 0
        hist_n, hist_a = queue_feats[mask_n], queue_feats[~mask_n]

        sim_n_n = torch.div(torch.matmul(q_n, torch.cat([k_n, hist_n], dim=0).T), self.T)
        sim_n_a = torch.div(torch.matmul(q_n, torch.cat([k_a, hist_a], dim=0).T), self.T)
        sim_a_n = torch.div(torch.matmul(q_a, torch.cat([k_n, hist_n], dim=0).T), self.T)

        sim_n_n_n_a = torch.cat([sim_n_n, sim_n_a], dim=1)
        sim_n_n_n_a_max, _ = torch.max(sim_n_n_n_a, dim=1, keepdim=True)
        sim_a_n_max, _ = torch.max(sim_a_n, dim=1, keepdim=True)

        sim_n_n = sim_n_n - sim_n_n_n_a_max.detach()
        sim_n_a = sim_n_a - sim_n_n_n_a_max.detach()
        sim_a_n = sim_a_n - sim_a_n_max.detach()

        exp_sim_n_a = torch.exp(sim_n_a)
        exp_sim_a_n = torch.exp(sim_a_n)

        loss_1 = sim_n_n - torch.log(torch.sum(exp_sim_n_a, dim=1, keepdim=True) + ESP)
        loss_1 = - torch.sum(torch.mean(loss_1, dim=1, keepdim=True))

        loss_2 = torch.mean(torch.log(torch.sum(exp_sim_a_n, dim=1, keepdim=True) + ESP))

        return loss_1 + loss_2
