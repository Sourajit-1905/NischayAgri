"""
Crop-level abstain and invalid-combo rejection.

Sits between HierarchicalHead's raw output and any downstream decision
logic. Two responsibilities:
  1. Decide whether the crop prediction itself is confident enough to act on.
  2. If so, mask the disease logits down to only the diseases that are
     actually valid for that crop (per label_space.py), so the final
     disease prediction can never disagree with the predicted crop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cropml.models.hierarchical_head import HeadOutput, HierarchicalHead


@dataclass(frozen=True)
class CropRejectionConfig:
    crop_confidence_threshold: float = 0.5


@dataclass(frozen=True)
class CropRejectionResult:
    crop_accepted: bool
    crop_idx: int
    crop_name: str
    crop_confidence: float
    # None when crop_accepted is False - no disease prediction is made
    # for an uncertain crop, per the plan's "abstain before predicting disease".
    disease_idx: int | None
    disease_class_id: str | None
    disease_confidence: float | None


def _build_valid_class_mask(head: HierarchicalHead) -> torch.Tensor:
    """
    (n_crops, n_classes) bool mask: mask[c, d] is True iff class d
    belongs to crop c. Built once from label_space.py's real crop field
    on each training class, not guessed from string parsing.
    """
    n_crops = len(head.crops)
    n_classes = len(head.classes)
    mask = torch.zeros(n_crops, n_classes, dtype=torch.bool)
    for class_idx, class_id in enumerate(head.classes):
        label = head.label_space.get(class_id)
        crop_idx = head.crop_to_idx[label.crop]
        mask[crop_idx, class_idx] = True
    return mask


class CropRejectionGate:
    """
    Stateful only in that it caches the valid-class mask for a given
    HierarchicalHead - cheap to construct, safe to keep around per model.
    """

    def __init__(self, head: HierarchicalHead, config: CropRejectionConfig | None = None) -> None:
        self.head = head
        self.config = config or CropRejectionConfig()
        self._valid_class_mask = _build_valid_class_mask(head)

    def apply(self, output: HeadOutput) -> list[CropRejectionResult]:
        """
        One CropRejectionResult per item in the batch.
        """
        batch_size = output.crop_logits.shape[0]
        crop_confidences, crop_indices = output.crop_probs.max(dim=-1)

        results: list[CropRejectionResult] = []
        for i in range(batch_size):
            crop_idx = int(crop_indices[i].item())
            crop_confidence = float(crop_confidences[i].item())
            crop_name = self.head.crops[crop_idx]
            accepted = crop_confidence >= self.config.crop_confidence_threshold

            if not accepted:
                results.append(
                    CropRejectionResult(
                        crop_accepted=False,
                        crop_idx=crop_idx,
                        crop_name=crop_name,
                        crop_confidence=crop_confidence,
                        disease_idx=None,
                        disease_class_id=None,
                        disease_confidence=None,
                    )
                )
                continue

            valid_mask = self._valid_class_mask[crop_idx]
            masked_logits = output.disease_logits[i].masked_fill(~valid_mask, float("-inf"))
            masked_probs = torch.softmax(masked_logits, dim=-1)
            disease_confidence, disease_idx_t = masked_probs.max(dim=-1)
            disease_idx = int(disease_idx_t.item())

            results.append(
                CropRejectionResult(
                    crop_accepted=True,
                    crop_idx=crop_idx,
                    crop_name=crop_name,
                    crop_confidence=crop_confidence,
                    disease_idx=disease_idx,
                    disease_class_id=self.head.classes[disease_idx],
                    disease_confidence=float(disease_confidence.item()),
                )
            )
            
        return results