import torch
from torch import optim, nn

import torch.distributed as dist
from torch import Tensor

import numpy as np
import random
import math
import os

import models
import moco
import losses


def init_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_models(args):
    if len(args.sources) == 1:
        encoder = models.SMSV(
            sources=args.sources,
            backbone="r3d_18",
            pretrained=True,
            return_features=False,
        )
    else:
        backbones = {}
        for source in args.sources:
            backbone = models.SMSV(
                sources=[source],
                backbone="r3d_18",
                pretrained=True,
                return_features=True,
            )
            ckpt_name = "best_dtc_pr.pth" if args.task == "detection" else "best_cls_pr.pth"
            backbone_ckpt_path = os.path.join(args.ckpt_dir_path, "single", f"SMSV_{args.classifier}_{source}_{args.task}", ckpt_name)
            backbone_ckpt = torch.load(backbone_ckpt_path, map_location="cpu")
            backbone.load_state_dict(backbone_ckpt["encoder"], strict=True)
            backbones[source] = backbone

        encoder = models.MMMV(
            backbones=backbones,
            fusion_method=args.fusion_method,
            fusion_steps=args.fusion_steps,
            mask_ratio=args.mask_ratio,
            backbone_out_channels=512,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone
        )

    in_dim, out_dim = encoder.out_dim, args.contrast_dim
    hidden_dim = min(max(in_dim // 2, out_dim * 2), in_dim)
    projector = models.MLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, normalize=True)
    model = moco.SuMoCo(base_encoder=encoder, mlp=projector, dim=out_dim, K=args.K, m=args.m)

    num_classes = 1 if args.task == "detection" else 9
    if args.classifier == "mlp":
        classifier = models.MLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=num_classes, normalize=False)
    else:
        classifier = models.MemoryBank(num_classes=num_classes, dim=in_dim)

    return model, classifier


def get_loss_functions(args):
    if args.task == "detection":
        # loss_fn_con = losses.DetConLoss(T=args.T)
        loss_fn_con = losses.SupConLoss(T=args.T)
        if args.classifier == "mlp":
            loss_fn_cls = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            loss_fn_cls = None

    else:
        loss_fn_con = losses.SupConLoss(T=args.T)
        if args.classifier == "mlp":
            loss_fn_cls = losses.FocalLoss()
        else:
            loss_fn_cls = None

    return loss_fn_con, loss_fn_cls


def get_model_optimizers(args, model):
    optimizer_con = optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler_con = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer_con,
        T_0=5,
        T_mult=1,
        eta_min=1e-6,
        verbose=True
    )

    return optimizer_con, scheduler_con


def get_head_optimizers(args, classifier):
    if args.classifier == "mlp":
        optimizer_cls = optim.Adam(
            params=classifier.parameters(),
            lr=min(0.01, args.learning_rate * 10),
            weight_decay=args.weight_decay
        )
        scheduler_cls = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer_cls,
            T_0=5,
            T_mult=1,
            eta_min=1e-5,
            verbose=True
        )
    else:
        optimizer_cls, scheduler_cls = None, None

    return optimizer_cls, scheduler_cls


def print_info(info: str, decorator: str = "=") -> None:
    _, num_cols = os.popen("stty size", "r").read().split()
    num_cols = int(num_cols)
    assert len(decorator) == 1
    print(decorator * num_cols)
    if len(info) < num_cols:
        print(f"{info:^{num_cols}s}")
    else:
        print(info)
    print(decorator * num_cols)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially, making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training), which means that the model params won't have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.

    Source: https://huggingface.co/transformers/v3.0.2/_modules/transformers/trainer.html
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def distributed_concat(tensor: Tensor, num_total_examples: int):
    """
    Concatenate all tensors across all devices.
    Source: https://huggingface.co/transformers/v3.0.2/_modules/transformers/trainer.html

    Args:
        tensor (Tensor): the tensor to be concatenated.
        num_total_examples (int): the total number of such tensor (i.e. the total batch size).

    Returns:
        concat (Tensor): the concatenated tensor.
    """
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    concat = concat[:num_total_examples]  # truncate the dummy elements added by SequentialDistributedSampler
    return concat


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
