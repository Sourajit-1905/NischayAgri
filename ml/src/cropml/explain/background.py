"""
Background-attention ratio and masking experiment.

background_attention_ratio: what fraction of a CAM's total attention
mass falls on background pixels (non-leaf region). A high ratio on a
correct prediction suggests a background shortcut.

masking_experiment: compares model confidence on the original image vs
a version with the background replaced by a neutral color, across a
batch of samples. A large confidence drop indicates background reliance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BackgroundAttentionResult:
    background_ratio: float     # fraction of CAM mass on background [0, 1]
    foreground_ratio: float     # 1 - background_ratio
    total_cam_mass: float


@dataclass(frozen=True)
class MaskingExperimentResult:
    mean_confidence_original: float
    mean_confidence_masked: float
    mean_confidence_drop: float     # original - masked; positive = background relied on
    n_samples: int


def background_attention_ratio(
    cam: torch.Tensor,          # (H, W) [0, 1], same spatial size as leaf_mask
    leaf_mask: torch.Tensor,    # (H, W) binary: 1=leaf, 0=background
) -> BackgroundAttentionResult:
    """Computes what fraction of CAM attention falls outside the leaf mask.
    leaf_mask should match the CAM's spatial resolution (resize if needed)."""
    if cam.shape != leaf_mask.shape:
        leaf_mask = F.interpolate(
            leaf_mask.unsqueeze(0).unsqueeze(0).float(),
            size=cam.shape, mode="nearest",
        ).squeeze()

    total = cam.sum().item()
    if total < 1e-8:
        return BackgroundAttentionResult(0.0, 0.0, 0.0)

    bg_mass = (cam * (1.0 - leaf_mask)).sum().item()
    fg_mass = (cam * leaf_mask).sum().item()
    return BackgroundAttentionResult(
        background_ratio=bg_mass / total,
        foreground_ratio=fg_mass / total,
        total_cam_mass=total,
    )


def masking_experiment(
    images: torch.Tensor,           # (N, C, H, W)
    leaf_masks: torch.Tensor,       # (N, 1, H, W) binary: 1=leaf, 0=background
    forward_fn: Callable[[torch.Tensor], torch.Tensor],  # images -> probs (N, n_classes)
    class_indices: list[int],
    fill_value: float = 0.5,        # neutral gray background fill
) -> MaskingExperimentResult:
    """Replaces background pixels with fill_value and measures confidence
    change. A large drop indicates the model relied on background features."""
    with torch.no_grad():
        orig_probs = forward_fn(images)
        masked_images = images * leaf_masks + fill_value * (1.0 - leaf_masks)
        masked_probs = forward_fn(masked_images)

    orig_confs = torch.tensor([orig_probs[i, c].item() for i, c in enumerate(class_indices)])
    masked_confs = torch.tensor([masked_probs[i, c].item() for i, c in enumerate(class_indices)])
    drops = orig_confs - masked_confs

    return MaskingExperimentResult(
        mean_confidence_original=float(orig_confs.mean().item()),
        mean_confidence_masked=float(masked_confs.mean().item()),
        mean_confidence_drop=float(drops.mean().item()),
        n_samples=len(images),
    )












