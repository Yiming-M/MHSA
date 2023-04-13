import torch
from torch.utils.data import DataLoader

import numpy as np

import yaml
from tqdm import tqdm
import os
import gc
import argparse

import metrics
import dataset
import utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="PyTorch DAD Training.")
parser.add_argument(
    "--data-dir-path",
    type=str,
    default=os.path.join(".", "data")
)
parser.add_argument(
    "--sources",
    type=str,
    nargs="+",
    required=True
)
parser.add_argument(
    "--task",
    type=str,
    choices=["detection", "classification"],
    default="detection"
)
parser.add_argument(
    "--fusion-method",
    type=str,
    default="Add",
)
parser.add_argument(
    "--fusion-steps",
    type=int,
    default=1,
)
parser.add_argument(
    "--classifier",
    type=str,
    choices=["mlp", "memory_bank"],
    default="mlp",
)
parser.add_argument(
    "--freeze-backbone",
    type=str,
    default="False"
)
parser.add_argument(
    "--mask-ratio",
    type=float,
    default=0.0
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.5
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-3
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=1e-3
)
parser.add_argument(
    "--ckpt-dir-path",
    type=str,
    default=os.path.join(".", "checkpoints"),
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
)
parser.add_argument(
    "--device",
    type=int,
    default=0,
)


def get_dataloaders(args):
    train_normal_batch_size = int(args.train_batch_size * 0.5)
    train_anomalous_batch_size = int(args.train_batch_size - train_normal_batch_size)
    train_normal_data = dataset.DAD(
        root=args.data_dir_path,
        sources=args.sources,
        task=args.task,
        split="train",
        category="normal",
        spatial_size=tuple(args.spatial_size),
        temporal_size=args.temporal_size,
        frames_per_clip=args.train_frames_per_clip,
        step_between_clips=args.train_step_between_clips,
        num_augs=2
    )
    train_normal_loader = DataLoader(
        dataset=train_normal_data,
        batch_size=train_normal_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0,
        drop_last=True,
        shuffle=True
    )
    train_anomalous_data = dataset.DAD(
        root=args.data_dir_path,
        sources=args.sources,
        task=args.task,
        split="train",
        category="anomalous",
        spatial_size=tuple(args.spatial_size),
        temporal_size=args.temporal_size,
        frames_per_clip=args.train_frames_per_clip,
        step_between_clips=args.train_step_between_clips,
        num_augs=2
    )
    train_anomalous_loader = DataLoader(
        dataset=train_anomalous_data,
        batch_size=train_anomalous_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0,
        drop_last=True,
        shuffle=True
    )

    if args.classifier == "memory_bank":
        train_both_data = dataset.DAD(
            root=args.data_dir_path,
            sources=args.sources,
            task=args.task,
            split="train",
            category="both",
            spatial_size=tuple(args.spatial_size),
            temporal_size=args.temporal_size,
            frames_per_clip=args.train_frames_per_clip,
            step_between_clips=args.train_frames_per_clip,
            num_augs=0
        )
        train_both_loader = DataLoader(
            dataset=train_both_data,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.num_workers > 0,
            drop_last=True,
            shuffle=False
        )
    else:
        train_both_loader = None

    test_data = dataset.DAD(
        root=args.data_dir_path,
        sources=args.sources,
        task=args.task,
        split="test",
        category="both",
        spatial_size=tuple(args.spatial_size),
        temporal_size=args.temporal_size,
        frames_per_clip=args.test_frames_per_clip,
        step_between_clips=args.test_step_between_clips,
        num_augs=0
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0,
        shuffle=False
    )
    return train_normal_loader, train_anomalous_loader, train_both_loader, test_loader


def main_worker(args: argparse.ArgumentParser):
    device = torch.device(args.device)
    utils.init_seeds(42)

    ckpt_name = "SMSV" if len(args.sources) == 1 else f"MMMV_{args.fusion_method}_{args.fusion_steps}"
    ckpt_name = ckpt_name + f"_mask_{args.mask_ratio}" if len(args.sources) > 1 and args.mask_ratio > 0 else ckpt_name
    ckpt_dir_path = os.path.join(args.ckpt_dir_path, ckpt_name + f"_{args.classifier}_{'_'.join(args.sources)}_{args.task}")
    os.makedirs(ckpt_dir_path, exist_ok=True)

    # --->>> define models --->>>
    model, classifier = utils.get_models(args)
    # <<<--- define models <<<---

    # --->>> define loss -->>>
    loss_fn_con, loss_fn_cls = utils.get_loss_functions(args)
    # <<<--- define loss <<<---

    # --->>> move to gpu --->>>
    model.to(device)
    classifier.to(device)
    loss_fn_con = loss_fn_con.to(device)
    if args.classifier == "mlp":
        loss_fn_cls = loss_fn_cls.to(device)
    # <<<--- move to gpu <<<---

    # --->>> define optimizer --->>>
    optimizer_con, scheduler_con = utils.get_model_optimizers(args, model)
    optimizer_cls, scheduler_cls = utils.get_head_optimizers(args, classifier)
    # <<<--- define optimizer <<<---

    # --->>> load checkpoints --->>>
    ckpt_path = os.path.join(ckpt_dir_path, "ckpt.pth")
    if os.path.isfile(ckpt_path):
        print("Found checkpoint. Loadind it.")
        ckpt = torch.load(ckpt_path, map_location=device)
        finish = ckpt["finish"]
        if finish:
            print("Training finished.")
        else:
            model.load_state_dict(ckpt["model"])
            classifier.load_state_dict(ckpt["classifier"])
            optimizer_con.load_state_dict(ckpt["optimizer_con"])
            scheduler_con.load_state_dict(ckpt["scheduler_con"])

            if args.classifier == "mlp":
                optimizer_cls.load_state_dict(ckpt["optimizer_cls"])
                scheduler_cls.load_state_dict(ckpt["scheduler_cls"])

            curr_epoch = ckpt["curr_epoch"]
            wait = ckpt["wait"]
    else:
        curr_epoch = 1
        wait = 0  # how many epochs the score has not been improved.
        finish = False  # the flag that indicates training has finished or not.

    score_path = os.path.join(ckpt_dir_path, "scores.pth")
    if os.path.isfile(score_path):
        score_ckpt = torch.load(score_path, map_location="cpu")
        costs = score_ckpt["costs"]
        hist_scores = score_ckpt["hist_scores"]
        best_scores = score_ckpt["best_scores"]
        eval_metrics = list(best_scores.keys())

    else:
        costs = {"con": [], "cls": []}
        eval_metrics = ["dtc_acc", "dtc_roc", "dtc_pr"] if args.task == "detection" else ["dtc_acc", "dtc_roc", "dtc_pr", "cls_acc", "cls_roc", "cls_pr"]
        hist_scores = {k: [] for k in eval_metrics}
        best_scores = {k: None for k in hist_scores.keys()}
    # <<<--- load checkpoints <<<---

    # --->>> define dataloaders and samplers --->>>
    train_normal_loader, train_anomalous_loader, train_both_loader, test_loader = get_dataloaders(args)
    # <<<--- define dataloaders and samplers <<<---

    while not finish:
        # training
        utils.print_info(
            f"SOURCES: {(', '.join(args.sources) + '.').ljust(45, ' ')}EPOCHS: {str(curr_epoch).zfill(len(str(args.max_epochs)))} / {args.max_epochs}",
            "*"
        )
        model.train(), classifier.train()

        cost_con, cost_cls = 0.0, 0.0

        num_batches = min(len(train_normal_loader), len(train_anomalous_loader))

        utils.print_info("Training", "-")
        for (n_imgs, n_labels), (a_imgs, a_labels) in tqdm(zip(train_normal_loader, train_anomalous_loader), total=num_batches):
            imgs = {source: torch.cat([n_imgs[source], a_imgs[source]], dim=0) for source in args.sources}
            labels = torch.cat([n_labels, a_labels], dim=0)

            imgs_q = {source: imgs[source][:, 0, :, :, :, :].to(device) for source in args.sources}
            imgs_k = {source: imgs[source][:, 1, :, :, :, :].to(device) for source in args.sources}
            labels = labels.to(device)

            with torch.enable_grad():
                q, k, queue_feats, queue_labels, hidden_feats = model(
                    x_q=imgs_q,
                    x_k=imgs_k,
                    y=labels
                )
                loss_con = loss_fn_con(q, k, labels, queue_feats, queue_labels)
                optimizer_con.zero_grad()
                loss_con.backward()
                optimizer_con.step()

            loss_con = loss_con.detach()
            cost_con += loss_con.item()

            if args.classifier == "mlp":
                with torch.enable_grad():
                    hidden_feats = hidden_feats.detach()
                    preds = classifier(hidden_feats)
                    if args.task == "detection":
                        labels = torch.unsqueeze(labels, dim=1).float()  # [bs, 1]
                    loss_cls = loss_fn_cls(preds, labels)
                    optimizer_cls.zero_grad()
                    loss_cls.backward()
                    optimizer_cls.step()

                    loss_cls = loss_cls.detach()
                    cost_cls += loss_cls.item()

        if args.classifier == "mlp":
            scheduler_cls.step()
            del preds, loss_cls

        del imgs, labels, imgs_q, imgs_k, q, k, queue_feats, queue_labels, hidden_feats, loss_con
        gc.collect()
        torch.cuda.empty_cache()

        if args.classifier == "memory_bank":
            model.eval(), classifier.eval()
            for imgs, labels in tqdm(train_both_loader):
                imgs = {source: imgs[source][:, 0, :, :, :, :].to(device) for source in args.sources}
                with torch.no_grad():
                    feats = model(x_q=imgs)

                if args.task == "detection":
                    classifier.update_memory(0, feats[labels == 1])
                else:
                    for i in range(9):
                        classifier.update_memory(i, feats[labels == i])

            del imgs, labels, feats
            gc.collect()
            torch.cuda.empty_cache()

        scheduler_con.step()

        cost_con /= num_batches
        if args.classifier == "mlp":
            cost_cls /= num_batches
            print(f"classification cost: {cost_cls:.4f}")
        else:
            cost_cls = None
        print(f"contrastive cost: \t {cost_con:.4f}")

        costs["con"].append(cost_con)
        costs["cls"].append(cost_cls)

        # evaluating
        model.eval(), classifier.eval()
        y_preds, y_trues = [], []
        utils.print_info("Evaluating", "-")
        for imgs, labels in tqdm(test_loader):
            imgs = {source: imgs[source][:, :, :, :, :].to(device) for source in args.sources}
            labels = labels.to(device)
            with torch.no_grad():
                feats = model(x_q=imgs)
                preds = classifier(feats)

                if args.classifier == "mlp":
                    if args.task == "classification":
                        preds = torch.softmax(preds, dim=1)
                    else:
                        preds = torch.sigmoid(preds)

            y_preds.append(preds.cpu().numpy())
            y_trues.append(labels.cpu().numpy())

        del imgs, labels, feats, preds
        gc.collect()
        torch.cuda.empty_cache()

        y_preds = np.concatenate(y_preds, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)

        if args.task == "detection":
            y_preds_ = y_preds[:, 0]
            y_trues_ = (y_trues == 1).astype(int)
            curr_scores = {metric: getattr(metrics, metric)(y_preds_, y_trues_) for metric in eval_metrics}
        else:
            mask_seen = y_trues != 9
            y_trues_seen = y_trues[mask_seen]
            y_preds_seen = y_preds[mask_seen, :]
            curr_scores = {}
            for metric in eval_metrics:
                if "cls" in metric:
                    curr_scores[metric] = getattr(metrics, metric)(y_preds_seen, y_trues_seen)
                else:
                    y_preds_ = y_preds[:, 0]
                    y_trues_ = (y_trues == 0).astype(int)
                    curr_scores[metric] = getattr(metrics, metric)(y_preds_, y_trues_)

        for k in curr_scores.keys():
            hist_scores[k].append(curr_scores[k])
            print(f"{k}\t curr:{curr_scores[k]}, best:{best_scores[k]}")
            if best_scores[k] is None or curr_scores[k] >= best_scores[k]:
                best_scores[k] = curr_scores[k]
                best_score_ckpt = {
                    "encoder": model.encoder_q.state_dict(),
                    "classifier": classifier.state_dict()
                }
                torch.save(best_score_ckpt, os.path.join(ckpt_dir_path, f"best_{k}.pth"))

        main_metric = "dtc_pr" if args.task == "detection" else "cls_pr"
        wait = wait + 1 if best_scores[main_metric] > curr_scores[main_metric] and curr_epoch >= args.min_epochs else 0
        if wait >= args.patience:
            utils.print_info("Early Stopping", "!")

        curr_epoch += 1
        if curr_epoch >= args.max_epochs or wait >= args.patience:
            finish = True

        ckpt = {"finish": finish}
        if not finish:
            ckpt["model"] = model.state_dict()
            ckpt["classifier"] = classifier.state_dict()
            ckpt["optimizer_con"] = optimizer_con.state_dict()
            ckpt["scheduler_con"] = scheduler_con.state_dict()

            if args.classifier == "mlp":
                ckpt["optimizer_cls"] = optimizer_cls.state_dict()
                ckpt["scheduler_cls"] = scheduler_cls.state_dict()

            ckpt["curr_epoch"] = curr_epoch
            ckpt["wait"] = wait

        score_ckpt = {
            "costs": costs,
            "hist_scores": hist_scores,
            "best_scores": best_scores
        }
        torch.save(ckpt, ckpt_path), torch.save(score_ckpt, score_path)


if __name__ == "__main__":
    args = parser.parse_args()
    args.freeze_backbone = args.freeze_backbone == "True"
    config_yaml_path = os.path.join(".", "config.yaml")
    with open(config_yaml_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    for k, v in config.items():
        if k not in args.__dict__.keys():
            setattr(args, k, v)

    args.sources.sort()
    main_worker(args)
