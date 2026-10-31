"""Crop -> disease hierarchical classification head.

Sits on top of a Backbone's embeddings. Two stages, trained jointly:
  1. crop head: embedding -> crop logits
  2. disease head: [embedding ; crop logits] -> disease logits (per crop's
     training-set disease vocabulary, via a shared layer over all diseases
     that appear for the TRAINING classes only - see label_space.py's
     training_classes, which excludes unseen_in_training and ambiguous).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cropml.data.label_space import LabelSpace, load_label_space


@dataclass(frozen=True)
class HeadOutput:
    crop_logits: torch.Tensor      # (batch, n_crops)
    disease_logits: torch.Tensor   # (batch, n_diseases)
    crop_probs: torch.Tensor       # (batch, n_crops), softmax(crop_logits)
    disease_probs: torch.Tensor    # (batch, n_diseases), softmax(disease_logits)


class HierarchicalHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        label_space: LabelSpace | None = None,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.label_space = label_space or load_label_space()

        training_classes = self.label_space.training_classes
        if not training_classes:
            raise ValueError("label_space has no training_classes - check label_space.yaml")

        # Crop vocabulary: unique crops among training classes only.
        self.crops: tuple[str, ...] = tuple(sorted({c.crop for c in training_classes}))
        self.crop_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.crops)}

        # Disease vocabulary: the FULL condition string (crop__condition's
        # condition part is NOT unique across crops - e.g. "healthy" repeats -
        # so we key by the full class_id to keep crop-specific diseases distinct.
        self.classes: tuple[str, ...] = tuple(c.id for c in training_classes)
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        n_crops = len(self.crops)
        n_diseases = len(self.classes)

        self.crop_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_crops),
        )
        self.disease_head = nn.Sequential(
            nn.Linear(embedding_dim + n_crops, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_diseases),
        )

    def forward(self, embeddings: torch.Tensor) -> HeadOutput:
        crop_logits = self.crop_head(embeddings)
        disease_input = torch.cat([embeddings, crop_logits], dim=-1)
        disease_logits = self.disease_head(disease_input)
        return HeadOutput(
            crop_logits=crop_logits,
            disease_logits=disease_logits,
            crop_probs=torch.softmax(crop_logits, dim=-1),
            disease_probs=torch.softmax(disease_logits, dim=-1),
        )

    def crop_index(self, crop_name: str) -> int:
        if crop_name not in self.crop_to_idx:
            raise KeyError(f"'{crop_name}' is not a training crop. Known: {self.crops}")
        return self.crop_to_idx[crop_name]

    def class_index(self, class_id: str) -> int:
        if class_id not in self.class_to_idx:
            raise KeyError(f"'{class_id}' is not a training class. Known: {self.classes}")
        return self.class_to_idx[class_id]

    def decode(self, crop_idx: int, disease_idx: int) -> tuple[str, str]:
        """Index -> (crop_name, class_id), for turning model output back
        into human-readable labels."""
        return self.crops[crop_idx], self.classes[disease_idx]