import torch
from torch import nn, Tensor

from typing import List, Any, Dict

from ..utils import _init_params


class Conv(nn.Module):
    """
    Feature fusion via squeeze-and-excitation.

    Args:
        - in_channels (int): the number of channels of each input tensor.
    """
    def __init__(
        self,
        sources: List[str],
        in_channels: int,
        fusion_steps: int = 1,
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
            self.conv = nn.Conv3d(in_channels=in_channels * len(self.sources), out_channels=in_channels, kernel_size=1)
        else:
            self.conv_top = nn.Conv3d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1)
            self.conv_front = nn.Conv3d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1)
            self.conv_out = nn.Conv3d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1)

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
            out = self.conv(x)
        else:
            x_top = torch.cat([x["top_IR"], x["top_depth"]], dim=1)
            x_top = self.conv_top(x_top)

            x_front = torch.cat([x["front_IR"], x["front_depth"]], dim=1)
            x_front = self.conv_front(x_front)

            x = torch.cat([x_top, x_front], dim=1)
            out = self.conv_out(x)

        out = self.avg_pool(out)
        out = self.flatten(out)
        return out
