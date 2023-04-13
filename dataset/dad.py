from torch import Tensor
from torch.utils.data import Dataset

import numpy as np

import pickle

import os
from copy import deepcopy

from typing import Union, List, Tuple, Dict, Optional

from .utils import TrainTransform, TestTransform, LabelTransform

rng = np.random.default_rng(42)

dad_means = {
    "front_depth": 0.244,
    "front_IR": 0.060,
    "top_depth": 0.196,
    "top_IR": 0.058
}

dad_stds = {
    "front_depth": 0.325,
    "front_IR": 0.105,
    "top_depth": 0.284,
    "top_IR": 0.108
}


class DAD(Dataset):
    def __init__(
        self,
        root: str,
        sources: Union[str, List[str]],
        task: str = "classification",
        split: str = "train",
        category: str = "both",
        temporal_size: int = 8,
        spatial_size: Union[int, Tuple[int, int]] = (112, 112),
        frames_per_clip: int = 64,
        step_between_clips: int = 1,
        num_augs: Optional[int] = None
    ) -> None:
        """
        Create a DAD PyTorch dataset.

        Args:
            - root (str): the root directory of all videos and pickle files.
            - sources (Union[str, List[str]]): the data source to use. Choose from `dad_sources`.
            - task (str): the task to perform ("classification" and "detection" only).
            - split (str, optional): the dataset split. Support "train" and "test".
            - category (str, optional): normal data or anomalous . Support "normal", "anomalous" and "both".
            - spatial_size (Union[int, Tuple[int, int]], optional): the spatial size of each returned clip.
            - temporal_size (int, optional): the temporal size of each returned clip.
            - frames_per_clip (int, optional): the number of frames each clip has before being sampled.
        """
        super().__init__()
        assert os.path.isdir(root)
        self.root = root

        sources = [sources] if isinstance(sources, str) else sources
        assert isinstance(sources, list)
        sources.sort()
        self.sources = sources

        assert task in ["detection", "classification"]
        self.task = task

        assert split in ["train", "test"]
        self.split = split
        self.pkl_file_path = os.path.join(root, f"{self.split}.pkl")
        assert os.path.exists(self.pkl_file_path)

        assert category in ["normal", "anomalous", "both"]
        self.category = category

        assert temporal_size > 0 and isinstance(temporal_size, int)
        self.temporal_size = temporal_size

        spatial_size = (spatial_size, spatial_size) if isinstance(spatial_size, int) else spatial_size
        assert (isinstance(spatial_size, tuple) and len(spatial_size) == 2)
        self.spatial_size = spatial_size

        assert frames_per_clip > 0 and isinstance(frames_per_clip, int)
        self.frames_per_clip = frames_per_clip

        assert step_between_clips > 0 and isinstance(step_between_clips, int)
        self.step_between_clips = step_between_clips

        self.label_transform = LabelTransform(task=self.task)

        with open(self.pkl_file_path, "rb") as f:
            pkl = deepcopy(pickle.load(f))

        assert set(self.sources).issubset(set(pkl["img_sources"]))

        if self.split == "train":
            assert set(pkl.keys()) == {"normal", "anomalous", "img_sources"}
            assert set(pkl["normal"].keys()) == set(pkl["anomalous"].keys()) == {"img_base_dirs", "img_names", "labels"}

            if self.category == "both":
                normal, anomalous = pkl["normal"], pkl["anomalous"]
                normal_clips, normal_labels = self.__make_data__(**normal)
                anomalous_clips, anomalous_labels = self.__make_data__(**anomalous)
                clips = normal_clips + anomalous_clips
                labels = normal_labels + anomalous_labels
            else:
                subset = pkl[self.category]
                clips, labels = self.__make_data__(**subset)

            indices = list(range(len(clips)))
            rng.shuffle(indices)
            self.clips = [clips[i] for i in indices]
            self.labels = [labels[i] for i in indices]
            self.clip_transform = TrainTransform(
                out_size=(temporal_size, *spatial_size),
                means=dad_means,
                stds=dad_stds,
                num_augs=num_augs
            )

        else:
            assert set(pkl.keys()) == {"img_base_dirs", "img_names", "labels", "img_sources"}
            img_base_dirs, img_names, labels = pkl["img_base_dirs"], pkl["img_names"], pkl["labels"]
            self.clips, self.labels = self.__make_data__(img_base_dirs, img_names, labels)

            self.clip_transform = TestTransform(
                out_size=(self.temporal_size, *self.spatial_size),
                means=dad_means,
                stds=dad_stds
            )

    def __len__(self) -> int:
        assert len(self.clips) == len(self.labels)
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[Union[Dict[str, Tensor], Tensor], Tensor]:
        clip_info = self.clips[idx]
        assert len(clip_info["img_names"]) == self.frames_per_clip

        img_base_dir = os.path.join(self.root, clip_info["img_base_dir"])
        clip = self.clip_transform(
            img_base_dir=img_base_dir,
            sources=self.sources,
            img_names=clip_info["img_names"]
        )  # train (dict): aug, c, t, h, w; test (dict): c, t, h, w

        label = self.label_transform(self.labels[idx])  # [1,]
        return clip, label

    def __make_data__(
        self,
        img_base_dirs: List[str],
        img_names: List[str],
        labels: List[str],
    ) -> Tuple[List[Dict], List[str]]:
        assert len(img_base_dirs) == len(img_names) == len(labels)
        clips, clip_labels = [], []

        counter1 = 0
        while counter1 < len(img_names):
            img_base_dir = img_base_dirs[counter1]
            label = labels[counter1]

            counter2 = counter1
            while counter2 < len(img_names) and img_base_dirs[counter2] == img_base_dir and labels[counter2] == label:
                counter2 += 1

            for i in range(counter1, counter2, self.step_between_clips):
                clip = {"img_base_dir": img_base_dir}
                start_idx, end_idx = i, min(i + self.frames_per_clip, counter2)
                clip["img_names"] = img_names[start_idx: end_idx]
                clip["img_names"] += clip["img_names"][-1:] * (self.frames_per_clip - len(clip["img_names"]))

                clips.append(clip)
                clip_labels.append(label)

            counter1 = counter2

        assert len(clips) == len(clip_labels)
        return clips, clip_labels
