import numpy as np
from copy import copy
from typing import List


class RandomReverse(object):
    """
    Reverse the input list.
    """
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.rng = np.random.default_rng()

    def __call__(self, vid: List) -> List:
        vid = copy(vid)
        assert isinstance(vid, list)
        if self.rng.random() <= self.p:
            vid.reverse()
        return vid


class UniformDownsample(object):
    """
    Downsample the input video.
    To reduce memory cost, this transform operates on the list of frame paths.
    """
    def __init__(self, num_frames: int = None, step: int = None) -> None:
        super().__init__()
        if num_frames is not None:
            assert isinstance(num_frames, int) and num_frames > 0 and step is None
        else:
            assert step is not None and isinstance(step, int) and step > 0

        self.num_frames = num_frames
        self.step = step
        self.rng = np.random.default_rng()

    def __call__(self, vid: List[str]) -> List[str]:
        assert isinstance(vid, list)

        num_all_frames = len(vid)
        num_sampled_frames = self.num_frames

        if num_sampled_frames is not None:
            assert num_sampled_frames <= num_all_frames, f"Downsampling cannot be executed. Tried to downsample {num_sampled_frames} frames from {num_all_frames} frames."
            if num_sampled_frames == 1:
                idx = int(self.rng.choice(a=num_all_frames, size=1))
                return [vid[idx]]
            else:
                num_intervals = num_sampled_frames - 1
                max_step = num_all_frames // num_intervals
                step = int(self.rng.choice(a=max_step, size=1)) + 1
                start_idx = int(self.rng.choice(a=range(0, max(1, num_all_frames - num_intervals * step)), size=1))
                end_idx = start_idx + num_intervals * step + 1
                ids = range(start_idx, end_idx, step)
                assert len(ids) == num_sampled_frames, f"Sampled: {len(ids)}; Needed: {num_sampled_frames}."
                return [vid[idx] for idx in ids]
        else:
            step = self.step
            ids = range(0, len(vid), step)

            return [vid[idx] for idx in ids]
