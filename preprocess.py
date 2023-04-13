import math
import pandas as pd
import pickle

import os
import shutil
from glob import glob

import numpy as np
from tqdm import tqdm

from typing import Union, Optional
import argparse

rng = np.random.default_rng(42)


def _get_img_id(img_path: str) -> int:
    img_name = os.path.basename(img_path)  # "./img_0.png" -> "img_0.png"
    img_name = img_name.split(".")[0]  # "img_0.png" -> "img_0"
    img_id = int(img_name.split("_")[-1])  # "img_0" -> 0
    return img_id


def _get_driver_id(driver_path: str) -> int:
    driver_name = os.path.basename(driver_path)  # "./Tester1" -> "Tester1"
    driver_id = int("".join(list(filter(str.isdigit, driver_name))))  # "Tester1" -> 1
    return driver_id


def prepare_DAD(
    root: str,
    split: str = "train",
    step_between_frames: int = 1,
    start: Optional[Union[float, int]] = None,
    end: Optional[Union[float, int]] = None,
    generate_new: bool = False
) -> None:
    """
    Prepare the DAD dataset.

    Args:
        root (str): the path of the DAD dataset, including the videos and labels.csv.
        start (Union[float, int]): for each video, frames within [0, start) will be trimmed.
            Defaults to None.
        end (Union[float, int]): for each video, frames within [end, -1] will be trimmed.
            Defaults to None
    """
    assert os.path.isdir(root) and os.path.exists(os.path.join(root, "labels.csv"))
    assert split in ["train", "test"]
    assert start is None or isinstance(start, int) or isinstance(start, float)
    assert end is None or isinstance(end, int) or isinstance(end, float)

    if generate_new:
        processed_dir_path = os.path.join(root, "processed")
        os.makedirs(processed_dir_path, exist_ok=True)

    keys = ["img_base_dirs", "img_names", "labels"]

    if split == "train":

        print("Processing training set.")
        normal_data, anomalous_data = {k: [] for k in keys}, {k: [] for k in keys}

        driver_paths = glob(os.path.join(root, "Tester*"))  # ["data_dir_path/Tester1", ...]
        driver_paths = filter(os.path.isdir, driver_paths)
        driver_names = [os.path.basename(driver_path) for driver_path in driver_paths]  # ["Tester1", ...]
        driver_names.sort(key=_get_driver_id)

        img_sources = None
        for driver in tqdm(driver_names):
            if generate_new:
                os.makedirs(
                    os.path.join(processed_dir_path, driver),
                    exist_ok=True
                )

            state_paths = glob(os.path.join(root, driver, "*"))  # ["data_dir_path/Tester1/drinking", ...]
            state_paths = filter(os.path.isdir, state_paths)
            state_names = [os.path.basename(state_path) for state_path in state_paths]  # ["drinking", ...]

            for state in state_names:
                if generate_new:
                    os.makedirs(
                        os.path.join(processed_dir_path, driver, state),
                        exist_ok=True
                    )

                source_paths = glob(os.path.join(root, driver, state, "*"))  # ["data_dir_path/Tester1/drinking/front_IR", ...]
                source_paths = filter(os.path.isdir, source_paths)
                source_names = [os.path.basename(source_path) for source_path in source_paths]  # ["front_IR", "top_IR", ...]
                source_names.sort()

                if img_sources is None:
                    img_sources = source_names
                else:
                    assert img_sources == source_names

                state_img_names = None
                for source in source_names:
                    img_paths = glob(os.path.join(root, driver, state, source, "img_*.png"))  # ["data_dir_path/Tester1/drinking/top_IR"/img_0.png", ...]
                    state_img_names_ = [os.path.basename(img_path) for img_path in img_paths]  # ["img_0.png", ...]
                    state_img_names_.sort(key=_get_img_id)

                    if start is None:
                        start_idx = 0
                    elif 0 <= start < 1:
                        start_idx = math.ceil(len(state_img_names_) * start)
                    else:
                        start_idx = min(max(int(start), 0), len(state_img_names_))

                    if end is None:
                        end_idx = len(state_img_names_)
                    elif 0 <= end < 1:
                        end_idx = math.floor(len(state_img_names_) * end)
                    else:
                        end_idx = max(min(int(end), len(state_img_names_)), 0)

                    assert end_idx > start_idx

                    state_img_names_ = state_img_names_[start_idx: end_idx: step_between_frames]
                    if state_img_names is None:
                        state_img_names = state_img_names_
                        if "normal" in state:
                            normal_data["img_base_dirs"] += [os.path.join(driver, state)] * len(state_img_names)
                            normal_data["img_names"] += state_img_names
                            normal_data["labels"] += [state] * len(state_img_names)
                        else:
                            anomalous_data["img_base_dirs"] += [os.path.join(driver, state)] * len(state_img_names)
                            anomalous_data["img_names"] += state_img_names
                            anomalous_data["labels"] += [state] * len(state_img_names)
                    else:
                        assert state_img_names == state_img_names_

                    if generate_new:
                        os.makedirs(
                            os.path.join(processed_dir_path, driver, state, source),
                            exist_ok=True
                        )
                        for img_name in state_img_names:
                            src_path = os.path.join(root, driver, state, source, img_name)
                            tgt_path = os.path.join(processed_dir_path, driver, state, source, img_name)
                            shutil.copy(src_path, tgt_path)

        train_data = {
            "normal": normal_data,
            "anomalous": anomalous_data,
            "img_sources": img_sources
        }
        pkl_path = os.path.join(processed_dir_path, "train.pkl") if generate_new else os.path.join(root, "train.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(train_data, f)

    else:
        print("Processing test set.")
        img_base_dirs, img_names, labels = [], [], []
        df_path = os.path.join(root, "labels.csv")
        df = pd.read_csv(df_path).dropna(how="all")
        img_sources = None

        for i in tqdm(range(len(df))):
            val, rec, start_idx, end_idx, state = df.iloc[i]
            start_idx, end_idx = int(start_idx), int(end_idx)

            if generate_new:
                os.makedirs(
                    os.path.join(processed_dir_path, val),
                    exist_ok=True
                )
                os.makedirs(
                    os.path.join(processed_dir_path, val, rec),
                    exist_ok=True
                )

            source_paths = glob(os.path.join(root, val, rec, "*"))
            source_paths = filter(os.path.isdir, source_paths)
            source_names = [os.path.basename(source_path) for source_path in source_paths]
            source_names.sort()

            if img_sources is None:
                img_sources = source_names
            else:
                assert source_names == img_sources

            state_img_names = None
            for source in source_names:
                img_paths = glob(os.path.join(root, val, rec, source, "img_*.png"))
                state_img_names_ = [os.path.basename(img_path) for img_path in img_paths]  # ["img_0.png", ...]
                state_img_names_.sort(key=_get_img_id)
                state_img_names_ = state_img_names_[start_idx: end_idx + 1]

                if start is None:
                    start_idx_ = 0
                elif 0 <= start < 1:
                    start_idx_ = math.ceil(len(state_img_names_) * start)
                else:
                    start_idx_ = min(max(int(start), 0), len(state_img_names_))

                if end is None:
                    end_idx_ = len(state_img_names_)
                elif 0 <= end < 1:
                    end_idx_ = math.floor(len(state_img_names_) * end)
                else:
                    end_idx_ = max(min(int(end), len(state_img_names_)), 0)

                assert end_idx_ > start_idx_

                state_img_names_ = state_img_names_[start_idx_: end_idx_: step_between_frames]
                if state_img_names is None:
                    state_img_names = state_img_names_
                    img_base_dirs += [os.path.join(val, rec)] * len(state_img_names)
                    img_names += state_img_names
                    labels += [state] * len(state_img_names)

                else:
                    assert state_img_names == state_img_names_

                if generate_new:
                    os.makedirs(
                        os.path.join(processed_dir_path, val, rec, source),
                        exist_ok=True
                    )
                    for img_name in state_img_names_:
                        src_path = os.path.join(root, val, rec, source, img_name)
                        tgt_path = os.path.join(processed_dir_path, val, rec, source, img_name)
                        shutil.copy(src_path, tgt_path)

        test_data = {
            "img_base_dirs": img_base_dirs,
            "img_names": img_names,
            "labels": labels,
            "img_sources": img_sources
        }
        pkl_path = os.path.join(processed_dir_path, "test.pkl") if generate_new else os.path.join(root, "test.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(test_data, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for preprocessing the DAD dataset.")

    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(".", "data"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train"
    )
    parser.add_argument(
        "--step-between-frames",
        type=int,
        default=2
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--generate-new",
        action="store_true"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    prepare_DAD(args.root, args.split, args.step_between_frames, args.start, args.end, args.generate_new)
