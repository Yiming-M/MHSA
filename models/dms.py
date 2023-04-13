import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Optional, Dict, Union

from .backbones import r3d_18, r2plus1d_18
from . import fusion
from .utils import _init_params


class SMSV(nn.Module):
    def __init__(
        self,
        sources: List[str],
        backbone: str = "r3d_18",
        pretrained: bool = True,
        return_features: bool = True,
    ) -> None:
        super().__init__()
        sources.sort()
        assert len(sources) == 1
        self.source = sources[0]

        assert backbone in ["r3d_18", "r2plus1d_18"]
        backbone = r3d_18 if backbone == "r3d_18" else r2plus1d_18
        backbone = backbone(pretrained=pretrained, in_channels=1)

        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.return_features = return_features
        if not self.return_features:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.flatten = nn.Flatten()
            self.out_dim = 512

    def forward(self, x: Union[Dict[str, Tensor], Tensor]) -> Tensor:
        if isinstance(x, dict):
            x_source = list(x.keys())
            assert x_source == [self.source]
            x = x[self.source]

        assert len(x.shape) == 5  # bs, c, t, h, w

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.return_features:
            x = self.avg_pool(x)
            x = self.flatten(x)
            x = F.normalize(x, p=2, dim=1)

        return x


class MMMV(nn.Module):
    def __init__(
        self,
        backbones: Dict[str, nn.Module],
        fusion_method: str,
        fusion_steps: int = 1,
        mask_ratio: Optional[float] = None,
        backbone_out_channels: int = 512,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        sources = list(backbones.keys())
        sources.sort()
        assert len(sources) > 1
        self.sources = sources

        self.freeze_backbone = freeze_backbone
        for source, backbone in backbones.items():
            if self.freeze_backbone:
                for param in self.parameters():
                    param.requires_grad = False
            setattr(self, f"backbone_{source}", backbone)

        assert fusion_method in ["Add", "Conv", "SE", "AFF", "MHSA"]
        fusion_method = getattr(fusion, fusion_method)(
            sources=sources,
            in_channels=backbone_out_channels,
            fusion_steps=fusion_steps,
            mask_ratio=mask_ratio,
            dropout=dropout,
        )
        _init_params(fusion_method)
        self.fusion = fusion_method
        self.out_dim = fusion_method.out_dim

    def get_features(self, source: str, x: Tensor) -> Tensor:
        backbone = getattr(self, f"backbone_{source}")
        if self.freeze_backbone:
            backbone.eval()
            with torch.no_grad():
                feat = backbone(x)
        else:
            feat = backbone(x)

        return feat

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        x_sources = list(x.keys())
        x_sources.sort()
        assert set(x_sources).issubset(self.sources)

        x = {source: self.get_features(source, x[source]) for source in x_sources}
        x = self.fusion(x)
        x = F.normalize(x, p=2, dim=1)

        return x
