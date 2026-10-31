from __future__ import annotations

import timm
import torch
from torch import nn

from cropml.models.backbone import Backbone, BackboneConfig


class EfficientNetB0Backbone(Backbone):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)
        self._timm_model = timm.create_model(
            config.timm_name, pretrained=config.pretrained, num_classes=0
        )
        if self._timm_model.num_features != config.embedding_dim:
            raise ValueError(
                f"config embedding_dim={config.embedding_dim} does not match "
                f"timm model's actual num_features={self._timm_model.num_features}"
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