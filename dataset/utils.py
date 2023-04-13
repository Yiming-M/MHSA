import torch
from torch import Tensor
from torchvision import transforms
import torchvision.transforms.functional as TF

from einops import rearrange
from copy import copy, deepcopy

import os
from PIL import Image
from typing import Sequence, Dict, Union, List, Tuple, Optional

from .spatial_transforms import Pepper, Salt, RandomCrop, RandomGaussianBlur
from .temporal_transforms import RandomReverse, UniformDownsample


class LabelTransform(object):
    """
    Transform the labels of DAD from str to torch.Tensor.
    """
    def __init__(self, task: str) -> None:
        super().__init__()
        assert task in ["detection", "classification"]
        self.task = task
        self.dictionary = {
            "normal_driving": 0,
            "adjusting_radio": 1,
            "drinking": 2,
            "messaging_left": 3,
            "messaging_right": 4,
            "reaching_behind": 5,
            "talking_with_passenger": 6,
            "talking_with_phone_left": 7,
            "talking_with_phone_right": 8,
            "unknown": 9
        }

    def __call__(self, orig_label: str) -> Tensor:
        if self.task == "classification":
            if "normal_driving" in orig_label:
                return torch.tensor(self.dictionary["normal_driving"])
            elif orig_label in self.dictionary.keys():
                return torch.tensor(self.dictionary[orig_label])
            else:
                return torch.tensor(self.dictionary["unknown"])
        else:
            if "normal_driving" in orig_label:
                return torch.tensor(1)
            else:
                return torch.tensor(0)


class LabelInverseTransform(object):
    """
    Transform the labels of DAD from str to torch.Tensor.
    """
    def __init__(self, task: str) -> None:
        super().__init__()
        assert task in ["detection", "classification"]
        self.task = task
        self.dictionary = {
            0: "normal driving",
            1: "adjusting radio",
            2: "drinking",
            3: "messaging (left)",
            4: "messaging (right)",
            5: "reaching behind",
            6: "talking with passenger",
            7: "talking with phone (left)",
            8: "talking with phone (right)",
            9: "unknown"
        }

    def __call__(self, label: Union[int, Tensor]) -> str:
        label = label.numpy() if isinstance(label, Tensor) else label
        label = int(label)

        if self.task == "classification":
            return self.dictionary[label]
        else:
            return "normal" if label == 1 else "anomalous"


def load_frames(img_base_dir, source, img_names):
    clip_source = []
    for img_name in img_names:
        img_path = os.path.join(img_base_dir, source, img_name)
        with Image.open(img_path) as img:
            img_ = deepcopy(img)
        clip_source.append(TF.to_tensor(copy(img_.convert("L"))))
    return clip_source


class TrainTransform(object):
    def __init__(
        self,
        out_size: Sequence[int],
        means: Dict[str, Union[float, Sequence[float]]],
        stds: Dict[str, Union[float, Sequence[float]]],
        num_augs: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert len(out_size) == 3
        temporal_size, spatial_size = out_size[0], out_size[1:]
        self.temporal_size = temporal_size
        self.spatial_size = spatial_size

        means_keys, stds_keys = list(means.keys()), list(stds.keys())
        means_keys.sort(), stds_keys.sort()
        assert means_keys == stds_keys
        self.data_sources = means_keys
        self.means = means
        self.stds = stds

        self.num_augs = num_augs if num_augs is not None else 0

        self.temporal_downsample = UniformDownsample(num_frames=self.temporal_size)
        self.spatial_downsample = transforms.Resize(size=self.spatial_size, interpolation=transforms.InterpolationMode.BILINEAR)

        if self.num_augs > 0:
            self.temporal_reverse = RandomReverse(p=0.5)
            self.spatial_transforms = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.5),
                transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
                transforms.RandomApply([RandomGaussianBlur(kernel_sizes=[1, 3, 5], sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomApply([RandomCrop(scales=(0.75, 1.0))], p=0.5),
            ])
            self.add_pepper_salt = transforms.Compose([
                transforms.RandomApply([Pepper(pepperness=5e-3)], p=0.5),
                transforms.RandomApply([Salt(saltness=5e-3)], p=0.5)
            ])

    def __call__(
        self,
        img_base_dir: str,
        sources: List[str],
        img_names: List[str]
    ) -> Dict[str, Tensor]:
        img_names = self.temporal_downsample(img_names)

        sources.sort()
        assert set(sources).issubset(self.data_sources)
        clip = {source: [] for source in sources}

        for source in sources:
            clip_source_orig = load_frames(img_base_dir, source, img_names)

            if self.num_augs > 0:
                for _ in range(self.num_augs):
                    clip_source_aug = self.temporal_reverse(copy(clip_source_orig))
                    clip_source_aug = torch.stack(clip_source_aug, dim=0)  # t, c, h, w
                    clip_source_aug = self.spatial_transforms(clip_source_aug)
                    clip_source_aug = self.add_pepper_salt(clip_source_aug)
                    clip_source_aug = self.spatial_downsample(clip_source_aug)
                    clip_source_aug = TF.normalize(clip_source_aug, mean=self.means[source], std=self.stds[source])
                    clip_source_aug = rearrange(clip_source_aug, "t c h w -> c t h w")

                    clip[source].append(clip_source_aug)

            else:
                clip_source_orig = torch.stack(clip_source_orig, dim=0)  # t, c, h, w
                clip_source_orig = self.spatial_downsample(clip_source_orig)  # t, c, h, w
                clip_source_orig = TF.normalize(clip_source_orig, mean=self.means[source], std=self.stds[source])
                clip_source_orig = rearrange(clip_source_orig, "t c h w -> c t h w")
                clip[source].append(clip_source_orig)

            clip[source] = torch.stack(clip[source], dim=0)  # num_augs (or 1), c, t, h, w

        return clip


class TestTransform(object):
    def __init__(
        self,
        out_size: Sequence[int],
        means: Dict[str, Union[float, Sequence[float]]],
        stds: Dict[str, Union[float, Sequence[float]]],
    ) -> None:
        super().__init__()
        assert len(out_size) == 3
        temporal_size, spatial_size = out_size[0], out_size[1:]
        self.temporal_size = temporal_size
        self.spatial_size = spatial_size

        means_keys, stds_keys = list(means.keys()), list(stds.keys())
        means_keys.sort(), stds_keys.sort()
        assert means_keys == stds_keys
        self.data_sources = means_keys
        self.means = means
        self.stds = stds

        self.temporal_downsample = UniformDownsample(num_frames=self.temporal_size)
        self.spatial_downsample = transforms.Resize(size=self.spatial_size, interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(
        self,
        img_base_dir: str,
        sources: List[str],
        img_names: List[str]
    ) -> Dict[str, Tensor]:
        img_names = self.temporal_downsample(img_names)

        sources.sort()
        assert set(sources).issubset(self.data_sources)
        clip = {}

        for source in sources:
            clip_source_orig = load_frames(img_base_dir, source, img_names)
            clip_source_orig = torch.stack(clip_source_orig, dim=0)  # t, c, h, w
            clip_source_orig = self.spatial_downsample(clip_source_orig)  # t, c, h, w
            clip_source_orig = TF.normalize(clip_source_orig, mean=self.means[source], std=self.stds[source])
            clip_source_orig = rearrange(clip_source_orig, "t c h w -> c t h w")
            clip[source] = clip_source_orig  # c, t, h, w

        return clip


def collate_fn(batch: List[Tuple[Dict[str, Tensor], Tensor]]) -> Tuple[List[Dict[str, Tensor]], List[Tensor]]:
    clips = {}
    labels = []
    for clip, label in batch:
        for k in clip.keys():
            if k not in clips.keys():
                clips[k] = [clip[k]]
            else:
                clips[k].append(clip[k])
        labels.append(label)

    for k in clips.keys():
        clips[k] = torch.stack(clips[k], dim=0)  # bs, c, t, h, w

    labels = torch.stack(labels, dim=0)  # bs

    return clips, labels
