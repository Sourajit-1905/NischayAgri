from __future__ import annotations

from cropml.models.backbone import (
    Backbone,
    BackboneConfig,
    BackboneConfigError,
    default_config_dir,
    load_backbone_config,
)
from cropml.models.efficientnet_b0 import EfficientNetB0Backbone
from cropml.models.mobilenet_v3 import MobileNetV3Backbone
from cropml.models.vit_small import ViTSmallBackbone

BACKBONES: dict[str, type[Backbone]] = {
    "mobilenet_v3": MobileNetV3Backbone,
    "efficientnet_b0": EfficientNetB0Backbone,
    "vit_small": ViTSmallBackbone,
}


def get_backbone(name: str, config_dir=None) -> Backbone:
    if name not in BACKBONES:
        raise BackboneConfigError(f"unknown backbone '{name}'. Available: {sorted(BACKBONES)}")
    config = load_backbone_config(name, config_dir=config_dir)
    return BACKBONES[name](config)


__all__ = [
    "BACKBONES",
    "Backbone",
    "BackboneConfig",
    "BackboneConfigError",
    "EfficientNetB0Backbone",
    "MobileNetV3Backbone",
    "ViTSmallBackbone",
    "default_config_dir",
    "get_backbone",
    "load_backbone_config",
]