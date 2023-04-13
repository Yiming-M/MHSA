import torch
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import os
import argparse

import models
import dataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Pass the directory of the checkpoints of all modalities.")
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(".", "checkpoints", "MMMV_MHSA_1_mlp_front_IR_front_depth_top_IR_top_depth_classification")
    )
    parser.add_argument(
        "--device",
        type=int,
        default=1
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device(args.device)

    sources = ["top_IR", "top_depth", "front_IR", "front_depth"]
    sources.sort()

    backbones = {}
    for source in sources:
        backbone = models.SMSV(
            sources=[source],
            backbone="r3d_18",
            pretrained=True,
            return_features=True,
        )
        backbones[source] = backbone

    encoder = models.MMMV(
        backbones=backbones,
        fusion_method="MHSA",
        fusion_steps=1,
        mask_ratio=0.0
    )
    classifier = models.MLP(out_dim=9, normalize=False)

    ckpt_dir_path = os.path.join(args.dir, "best_cls_acc.pth")
    ckpt = torch.load(ckpt_dir_path, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    classifier.load_state_dict(ckpt["classifier"])

    test_data = dataset.DAD(
        root="data",
        sources=sources,
        task="classification",
        split="test",
        category="both",
        spatial_size=(112, 112),
        temporal_size=8,
        frames_per_clip=8,
        step_between_clips=4,
        num_augs=0
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=32,
        num_workers=1,
        shuffle=False
    )

    encoder, classifier = encoder.to(args.device), classifier.to(args.device)
    encoder.eval(), classifier.eval()
    y_preds, y_trues = [], []

    for imgs, labels in tqdm(test_loader):
        imgs = {source: imgs[source][:, :, :, :, :].to(args.device) for source in sources}
        labels = labels.to(args.device)
        with torch.no_grad():
            feats = encoder(imgs)
            preds = classifier(feats)
            preds = torch.softmax(preds, dim=1)

        y_preds.append(preds.cpu().numpy())
        y_trues.append(labels.cpu().numpy())

    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)

    mask_seen = y_trues != 9
    y_trues_seen = y_trues[mask_seen]
    y_preds_seen = np.argmax(y_preds[mask_seen, :], axis=1)
    cm = confusion_matrix(y_trues_seen, y_preds_seen, normalize="true")
    np.save("cm.npy", cm)
