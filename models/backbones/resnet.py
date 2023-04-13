import torch
from torch import nn, Tensor

from torch.hub import load_state_dict_from_url

from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, OrderedDict, Dict

from ..utils import _log_api_usage_once


model_urls = {
    "r3d": "https://download.pytorch.org/models/r3d_18-b3b3357e.pth",
    "r(2+1)d": "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth"
}


class Conv3DSimple(nn.Conv3d):
    def __init__(
        self, in_planes: int, out_planes: int, midplanes: Optional[int] = None, stride: int = 1, padding: int = 1
    ) -> None:

        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, stride, stride


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = 1, padding: int = 1) -> None:
        super().__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes, out_planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False
            ),
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, stride, stride


class Conv3DNoTemporal(nn.Conv3d):
    def __init__(
        self, in_planes: int, out_planes: int, midplanes: Optional[int] = None, stride: int = 1, padding: int = 1
    ) -> None:

        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return 1, stride, stride


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super().__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride), nn.BatchNorm3d(planes), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(conv_builder(planes, planes, midplanes), nn.BatchNorm3d(planes))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:

        super().__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False), nn.BatchNorm3d(planes), nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride), nn.BatchNorm3d(planes), nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution"""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__(
            nn.Conv3d(in_channels, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class VideoResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv_makers: Sequence[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
        layers: List[int],
        stem: Callable[..., nn.Module],
        in_channels: int,
        zero_init_residual: bool = False,
        **kwargs: Any,
    ) -> None:
        """Generic resnet video generator.

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): resnet building block
            conv_makers (List[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]]): generator
                function for each layer
            layers (List[int]): number of blocks per layer
            stem (Callable[..., nn.Module]): module specifying the ResNet stem.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super().__init__()
        _log_api_usage_once(self)
        self.inplanes = 64

        self.stem = stem(in_channels=in_channels)

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr, arg-type]

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv_builder: Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)


def _video_resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    conv_makers: Sequence[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]],
    layers: List[int],
    stem: Callable[..., nn.Module],
    **kwargs: Any,
) -> VideoResNet:
    model = VideoResNet(block, conv_makers, layers, stem, **kwargs)
    return model


def r3d_18(pretrained: bool = True, in_channels: int = 1, **kwargs: Any) -> VideoResNet:
    model = _video_resnet(
        block=BasicBlock,
        conv_makers=[Conv3DSimple] * 4,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        in_channels=in_channels,
        **kwargs,
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["r3d"])
        state_dict = _modify_weights(state_dict, in_channels)
        model.load_state_dict(state_dict, strict=True)

    return model


def r2plus1d_18(pretrained: bool = True, in_channels: int = 1, **kwargs: Any) -> VideoResNet:
    model = _video_resnet(
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[2, 2, 2, 2],
        stem=R2Plus1dStem,
        in_channels=in_channels,
        **kwargs,
    )

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["r(2+1)d"])
        state_dict = _modify_weights(state_dict, in_channels)
        model.load_state_dict(state_dict, strict=True)

    return model


def _modify_weights(weights: OrderedDict, in_channels: int) -> Dict:
    new_weights = {}
    for k, v in weights.items():
        if "fc" not in k:
            new_weights[k] = v

    if in_channels != 3:
        new_weights["stem.0.weight"] = torch.cat([torch.sum(new_weights["stem.0.weight"], dim=1, keepdim=True)] * in_channels, dim=1) / in_channels
    return new_weights
