import torch
from torch import nn, Tensor

from typing import List, Any, Dict

from ..utils import _init_params


class SE(nn.Module):
    """
    Feature fusion via squeeze-and-excitation.

    Args:
        - in_channels (int): the number of channels of each input tensor.
        - r (float): the reducing factor for squeeze-and-excitation.
    """
    def __init__(
        self,
        sources: List[str],
        in_channels: int,
        fusion_steps: int = 1,
        r: float = 4.0,
        **kwargs: Any
    ) -> None:
        super().__init__()
        sources.sort()
        self.sources = sources

        assert fusion_steps in [1, 2]
        if len(self.sources) in [1, 2]:
            self.fusion_steps = 1
        else:
            assert len(self.sources) == 4
            self.fusion_steps = fusion_steps

        if self.fusion_steps == 1:
            in_channels_ = int(in_channels * len(sources))
            mid_channels = int(in_channels_ // r)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(in_channels=in_channels_, out_channels=mid_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, in_channels_, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(in_channels_),
                nn.Sigmoid()
            )
        else:
            in_channels_ = int(in_channels * 2)
            mid_channels = int(in_channels_ // r)
            self.se_top = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(in_channels=in_channels_, out_channels=mid_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, in_channels_, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(in_channels_),
                nn.Sigmoid()
            )
            self.se_front = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(in_channels=in_channels_, out_channels=mid_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, in_channels_, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(in_channels_),
                nn.Sigmoid()
            )
            self.se_out = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(in_channels=in_channels_, out_channels=mid_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, in_channels_, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(in_channels_),
                nn.Sigmoid()
            )

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.out_dim = in_channels

        _init_params(self)

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        x_sources = list(x.keys())
        x_sources.sort()
        assert x_sources == self.sources

        if self.fusion_steps == 1:
            x = torch.cat([x[source] for source in self.sources], dim=1)
            w = self.se(x)
            x = torch.chunk(x, len(self.sources), dim=1)
            w = torch.chunk(w, len(self.sources), dim=1)

            out = sum([x_ * w_ for x_, w_ in zip(x, w)])
        else:
            x_top = torch.cat([x["top_IR"], x["top_depth"]], dim=1)
            w_top = self.se_top(x_top)
            w_top = torch.chunk(w_top, 2, dim=1)
            x_top = w_top[0] * x["top_IR"] + w_top[1] * x["top_depth"]

            x_front = torch.cat([x["front_IR"], x["front_depth"]], dim=1)
            w_front = self.se_front(x_front)
            w_front = torch.chunk(w_front, 2, dim=1)
            x_front = w_front[0] * x["front_IR"] + w_front[1] * x["front_depth"]

            out = torch.cat([x_top, x_front], dim=1)
            w_out = self.se_out(out)
            w_out = torch.chunk(w_out, 2, dim=1)
            out = w_out[0] * x_top + w_out[1] * x_front

        out = self.avg_pool(out)
        out = self.flatten(out)
        return out
