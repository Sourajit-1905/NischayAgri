"""
Common interface for all backbones (MobileNetV3, EfficientNet-B0, ViT-small).

Backbones return EMBEDDINGS only, never class logits - classification heads
live in hierarchical_head.py. This keeps the same backbone reusable for
plain classification, MC-dropout, and feature-based OOD scoring (Mahalanobis).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "model"

class BackboneConfigError(ValueError):
    """
    The backbone config is invalid or names an unsupported backbone.
    """


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    timm_name: str
    pretrained: bool
    input_size: int
    dropout: float
    embedding_dim: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    freeze_backbone_epochs: int
    license: str

    @classmethod
    def from_yaml(cls, path: Path | str) -> BackboneConfig:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BackboneConfig:
        required = (
            "name", "timm_name", "pretrained", "input_size", "dropout",
            "embedding_dim", "batch_size", "learning_rate", "weight_decay",
            "freeze_backbone_epochs", "license",
        )  # fmt: skip
        missing = [k for k in required if k not in raw]
        if missing:
            raise BackboneConfigError(f"missing required config keys: {missing}")
        return cls(
            name=str(raw["name"]),
            timm_name=str(raw["timm_name"]),
            pretrained=bool(raw["pretrained"]),
            input_size=int(raw["input_size"]),
            dropout=float(raw["dropout"]),
            embedding_dim=int(raw["embedding_dim"]),
            batch_size=int(raw["batch_size"]),
            learning_rate=float(raw["learning_rate"]),
            weight_decay=float(raw["weight_decay"]),
            freeze_backbone_epochs=int(raw["freeze_backbone_epochs"]),
            license=str(raw["license"]),
        )


class Backbone(nn.Module, ABC):
    """
    Common interface every concrete backbone (step 4) implements.
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self._mc_dropout_enabled = False

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (batch, 3, H, W) -> embeddings: (batch, embedding_dim).
        """

    @abstractmethod
    def _dropout_modules(self) -> list[nn.Dropout]:
        """
        All nn.Dropout submodules this backbone actually contains.
        Used by enable_mc_dropout/disable_mc_dropout to toggle them without
        affecting other modules (e.g. BatchNorm) that .train()/.eval() would
        otherwise also flip.
        """

    @abstractmethod
    def freeze_backbone(self) -> None:
        """
        Freeze all parameters except the final feature-projection layer,
        per this backbone's freeze_backbone_epochs warmup strategy.
        """

    @abstractmethod
    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for full fine-tuning."""

    def enable_mc_dropout(self) -> None:
        """
        Keep dropout layers active even when the rest of the model is in
        eval() mode - required for MC-dropout (Phase 2.3 step 11) to sample
        multiple stochastic forward passes at inference time.
        """
        self._mc_dropout_enabled = True
        for d in self._dropout_modules():
            d.train()

    def disable_mc_dropout(self) -> None:
        """
        Restore normal eval-mode behavior (dropout off during inference).
        """
        self._mc_dropout_enabled = False
        for d in self._dropout_modules():
            d.eval()

    def eval(self) -> Backbone:  # type: ignore[override]
        """
        Override: eval() must not silently undo enable_mc_dropout(). The
        base nn.Module.eval() would otherwise flip every submodule, including
        dropout, back off.
        """
        result = super().eval()
        if self._mc_dropout_enabled:
            for d in self._dropout_modules():
                d.train()
        return result  # type: ignore[return-value]

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim


def default_config_dir() -> Path:
    return _DEFAULT_CONFIG_DIR


def load_backbone_config(name: str, config_dir: Path | None = None) -> BackboneConfig:
    path = (config_dir or default_config_dir()) / f"{name}.yaml"
    if not path.exists():
        raise BackboneConfigError(f"no config found for backbone '{name}' at {path}")
    return BackboneConfig.from_yaml(path)