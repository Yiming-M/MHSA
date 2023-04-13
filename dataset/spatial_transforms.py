import torch
from torch import Tensor
import torchvision.transforms.functional as TF

import numpy as np

from PIL import Image

from typing import Union, Sequence, Tuple


class Salt(object):
    """
    Add salt to the input image.
    """
    def __init__(self, saltness: Union[int, float] = 1e-2) -> None:
        super().__init__()
        self.saltness = saltness
        self.rng = np.random.default_rng()

    def __call__(self, imgs: Union[Tensor, Image.Image]) -> Union[Tensor, Image.Image]:
        assert isinstance(imgs, Tensor) or isinstance(imgs, Image.Image)
        type_orig = "tensor" if isinstance(imgs, Tensor) else "pil"
        if isinstance(imgs, Image.Image):
            imgs = TF.to_tensor(imgs)

        orig_shape = imgs.shape
        imgs = torch.reshape(imgs, shape=[-1])

        if int(self.saltness) != self.saltness:
            num_salts = int(self.saltness * len(imgs))
        else:
            assert self.saltness <= len(imgs)
            num_salts = int(self.saltness)

        indices = self.rng.choice(a=len(imgs), size=num_salts, replace=False)
        imgs[indices] = 0.0  # add salt.
        imgs = torch.reshape(imgs, shape=orig_shape)

        imgs = imgs if type_orig == "tensor" else TF.to_pil_image(imgs)
        return imgs


class Pepper(object):
    """
    Add salt to the input image.
    """
    def __init__(self, pepperness: Union[int, float] = 1e-2) -> None:
        super().__init__()
        self.pepperness = pepperness
        self.rng = np.random.default_rng()

    def __call__(self, imgs: Union[Tensor, Image.Image]) -> Union[Tensor, Image.Image]:
        assert isinstance(imgs, Tensor) or isinstance(imgs, Image.Image)
        type_orig = "tensor" if isinstance(imgs, Tensor) else "pil"
        if isinstance(imgs, Image.Image):
            imgs = TF.to_tensor(imgs)

        orig_shape = imgs.shape
        imgs = torch.reshape(imgs, shape=[-1])

        if int(self.pepperness) != self.pepperness:
            num_peppers = int(self.pepperness * len(imgs))
        else:
            assert self.pepperness <= len(imgs)
            num_peppers = int(self.pepperness)

        indices = self.rng.choice(a=len(imgs), size=num_peppers, replace=False)
        imgs[indices] = 1.0  # add pepper.
        imgs = torch.reshape(imgs, shape=orig_shape)

        imgs = imgs if type_orig == "tensor" else TF.to_pil_image(imgs)
        return imgs


class RandomCrop(object):
    """
    Apply torchvision.transforms.RandomCrop or a corner crop with a randomly selected scale.
    """
    def __init__(self, scales: Tuple[float, float] = (0.75, 1.0)) -> None:
        super().__init__()
        assert isinstance(scales, tuple) and len(scales) == 2
        assert 0. < scales[0] <= scales[1] <= 1.
        self.scales = scales
        self.corners = ["tl", "tr", "bl", "br"]
        self.rng = np.random.default_rng()

    def __generate_crop_size__(self, img_h: int, img_w: int) -> Tuple[int, int]:
        scale_h, scale_w = self.scales[0] + self.rng.random() * (self.scales[1] - self.scales[0]), self.scales[0] + self.rng.random() * (self.scales[1] - self.scales[0])
        crop_h, crop_w = int(img_h * scale_h), int(img_w * scale_w)
        return (crop_h, crop_w)

    def __random_crop__(self, imgs: Tensor) -> Tensor:
        img_h, img_w = imgs.shape[-2:]
        crop_h, crop_w = self.__generate_crop_size__(img_h, img_w)

        top = self.rng.choice(img_h - crop_h)
        left = self.rng.choice(img_w - crop_w)

        return TF.crop(imgs, top=top, left=left, height=crop_h, width=crop_w)

    def __corner_crop__(self, corner: str, imgs: Tensor) -> Tensor:
        assert corner in self.corners
        img_h, img_w = imgs.shape[-2:]
        crop_h, crop_w = self.__generate_crop_size__(img_h, img_w)

        if corner == "tl":
            top, left = 0, 0
        elif corner == "tr":
            top, left = 0, img_w - crop_w
        elif corner == "bl":
            top, left = img_h - crop_h, 0
        else:
            top, left = img_h - crop_h, img_w - crop_w

        return TF.crop(imgs, top=top, left=left, height=crop_h, width=crop_w)

    def __call__(self, imgs: Union[Tensor, Image.Image]) -> Union[Tensor, Image.Image]:
        assert isinstance(imgs, Tensor) or isinstance(imgs, Image.Image)
        type_orig = "tensor" if isinstance(imgs, Tensor) else "pil"
        if isinstance(imgs, Image.Image):
            imgs = TF.to_tensor(imgs)

        p = self.rng.random()
        if 0 <= p < 0.2:
            imgs = self.__random_crop__(imgs)
        elif 0.2 <= p < 0.4:
            imgs = self.__corner_crop__(self.corners[0], imgs)
        elif 0.4 <= p < 0.6:
            imgs = self.__corner_crop__(self.corners[1], imgs)
        elif 0.6 <= p < 0.8:
            imgs = self.__corner_crop__(self.corners[2], imgs)
        else:
            imgs = self.__corner_crop__(self.corners[3], imgs)

        imgs = imgs if type_orig == "tensor" else TF.to_pil_image(imgs)
        return imgs


class RandomGaussianBlur(object):
    """
    Apply Gaussian blur with randomly selected kernel_size and sigma.
    """
    def __init__(
        self,
        kernel_sizes: Union[Sequence[int], int] = [1, 3, 5],
        sigma: Union[Tuple[float, float], float] = (0.1, 2.0)
    ) -> None:
        super().__init__()

        kernel_sizes = [kernel_sizes] if isinstance(kernel_sizes, int) else kernel_sizes
        assert len(kernel_sizes) > 1 and min(kernel_sizes) > 0
        for kernel_size in kernel_sizes:
            assert kernel_size % 2 == 1

        sigma = (sigma,) if isinstance(sigma, float) else sigma
        assert isinstance(sigma, tuple) and 1 <= len(sigma) <= 2
        assert min(sigma) > 0

        self.kernel_sizes = kernel_sizes
        self.sigma = sigma
        self.rng = np.random.default_rng()

    def __call__(self, imgs: Union[Tensor, Image.Image]) -> Union[Tensor, Image.Image]:
        assert isinstance(imgs, Tensor) or isinstance(imgs, Image.Image)
        type_orig = "tensor" if isinstance(imgs, Tensor) else "pil"
        if isinstance(imgs, Image.Image):
            imgs = TF.to_tensor(imgs)

        kernel_x, kernel_y = self.rng.choice(self.kernel_sizes, size=2)
        if len(self.sigma) == 1:
            sigma_x, sigma_y = self.sigma, self.sigma
        else:
            sigma_x, sigma_y = self.sigma[0] + self.rng.random() * (self.sigma[1] - self.sigma[0]), self.sigma[0] + self.rng.random() * (self.sigma[1] - self.sigma[0])

        imgs = TF.gaussian_blur(imgs, kernel_size=(kernel_x, kernel_y), sigma=(sigma_x, sigma_y))
        imgs = imgs if type_orig == "tensor" else TF.to_pil_image(imgs)
        return imgs
