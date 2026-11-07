"""
Export a trained backbone+head to ONNX for deployment in apps/api.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from cropml.models.backbone import Backbone
from cropml.models.hierarchical_head import HierarchicalHead


@dataclass(frozen=True)
class ExportConfig:
    input_size: int = 224
    opset_version: int = 18
    dynamic_batch: bool = True


class _CombinedModel(nn.Module):
    """
    Wraps backbone+head into one nn.Module for clean ONNX export.
    Returns (crop_logits, disease_logits) as a flat tuple - ONNX handles
    named outputs via the export call's output_names argument.
    """

    def __init__(self, backbone: Backbone, head: HierarchicalHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.backbone(images)
        out = self.head(embeddings)
        return out.crop_logits, out.disease_logits


def export_to_onnx(
    backbone: Backbone,
    head: HierarchicalHead,
    output_path: Path | str,
    config: ExportConfig | None = None,
) -> Path:
    """
    Exports backbone+head to ONNX. Returns the written path.
    """
    config = config or ExportConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = _CombinedModel(backbone, head)
    model.eval()

    dummy = torch.zeros(1, 3, config.input_size, config.input_size)
    dynamic_axes: dict[str, dict[int, str]] | None = None
    if config.dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch_size"},
            "crop_logits": {0: "batch_size"},
            "disease_logits": {0: "batch_size"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            opset_version=config.opset_version,
            input_names=["images"],
            output_names=["crop_logits", "disease_logits"],
            dynamic_axes=dynamic_axes,
        )

    return output_path