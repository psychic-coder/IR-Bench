"""CNN backbones (torchvision): ResNet50 and EfficientNet-B0 conv feature maps + pooling."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import models

from irbench.features.base import register_extractor
from irbench.features.pooling import CONV_POOLINGS, pool_conv
from irbench.features.torch_base import TorchExtractor


@register_extractor
class ResNet50Extractor(TorchExtractor):
    """ResNet50 (ImageNet) truncated after ``layer4`` (2048-d) or ``layer3`` (1024-d)."""

    name = "resnet50"
    family = "cnn"
    poolings = CONV_POOLINGS
    default_pooling = "gem"
    supports_multiscale = True
    description = "ResNet50 ImageNet, last conv block feature map + GAP/GeM/R-MAC"

    _LAYER_DIMS = {"layer3": 1024, "layer4": 2048}

    def __init__(self, layer: str = "layer4", weights: str = "IMAGENET1K_V2", **kwargs: Any) -> None:
        if layer not in self._LAYER_DIMS:
            raise ValueError(f"layer must be one of {sorted(self._LAYER_DIMS)}")
        self.layer = layer
        self.weights = weights
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        return self._LAYER_DIMS[self.layer]

    def _extra_config(self) -> dict[str, Any]:
        return {**super()._extra_config(), "layer": self.layer, "weights": self.weights}

    def _build_model(self) -> nn.Module:
        weights = models.ResNet50_Weights[self.weights] if self.pretrained else None
        net = models.resnet50(weights=weights)
        stages: list[nn.Module] = [
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
        ]
        if self.layer == "layer4":
            stages.append(net.layer4)
        return nn.Sequential(*stages)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        return pool_conv(self.model(batch), self.pooling, self.gem_p)


@register_extractor
class EfficientNetB0Extractor(TorchExtractor):
    """EfficientNet-B0 (ImageNet) final 1x1 conv feature map (1280-d)."""

    name = "efficientnet_b0"
    family = "cnn"
    poolings = CONV_POOLINGS
    default_pooling = "gem"
    supports_multiscale = True
    description = "EfficientNet-B0 ImageNet, final conv feature map + GAP/GeM/R-MAC"

    def __init__(self, weights: str = "IMAGENET1K_V1", **kwargs: Any) -> None:
        self.weights = weights
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        return 1280

    def _extra_config(self) -> dict[str, Any]:
        return {**super()._extra_config(), "weights": self.weights}

    def _build_model(self) -> nn.Module:
        weights = models.EfficientNet_B0_Weights[self.weights] if self.pretrained else None
        return models.efficientnet_b0(weights=weights).features

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        return pool_conv(self.model(batch), self.pooling, self.gem_p)
