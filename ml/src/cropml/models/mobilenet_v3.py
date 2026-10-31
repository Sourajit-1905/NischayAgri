from __future__ import annotations

import timm
import torch
from torch import nn

from cropml.models.backbone import Backbone, BackboneConfig


class MobileNetV3Backbone(Backbone):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)
        self._timm_model = timm.create_model(
            config.timm_name, pretrained=config.pretrained, num_classes=0
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.input_size, config.input_size)
            actual_dim = self._timm_model(dummy).shape[-1]
        if actual_dim != config.embedding_dim:
            raise ValueError(
                f"config embedding_dim={config.embedding_dim} does not match "
                f"the model's actual forward-pass output dim={actual_dim} "
                f"(verified via a live forward pass, not the .num_features attribute, "
                f"which can be misleading for architectures with a head conv after "
                f"the last feature block)"
            )
        self._dropout = nn.Dropout(p=config.dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self._timm_model(images)
        return self._dropout(features)

    def _dropout_modules(self) -> list[nn.Dropout]:
        return [self._dropout]

    def freeze_backbone(self) -> None:
        for param in self._timm_model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self._timm_model.parameters():
            param.requires_grad = True