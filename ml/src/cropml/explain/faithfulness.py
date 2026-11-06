"""
Deletion and insertion faithfulness tests for CAM explanations.

Deletion: progressively mask out the most-attended pixels (by CAM score,
highest first) and measure how fast the model's confidence drops.
A faithful explanation should cause a sharp early drop.

Insertion: progressively reveal pixels starting from the most-attended.
A faithful explanation should cause a sharp early rise.

Both return AUC over the curve - lower deletion AUC and higher insertion
AUC indicate a more faithful explanation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FaithfulnessResult:
    deletion_auc: float     # lower is better (confidence drops quickly)
    insertion_auc: float    # higher is better (confidence rises quickly)
    deletion_curve: list[float]
    insertion_curve: list[float]
    n_steps: int


def _make_steps(cam: torch.Tensor, n_steps: int) -> list[torch.Tensor]:
    """Returns n_steps binary masks, each revealing more of the image
    in order of descending CAM importance."""
    H, W = cam.shape
    flat = cam.reshape(-1)
    order = flat.argsort(descending=True)
    step_size = max(1, len(flat) // n_steps)
    masks = []
    for i in range(n_steps):
        mask = torch.zeros_like(flat)
        mask[order[:step_size * (i + 1)]] = 1.0
        masks.append(mask.reshape(H, W))
    return masks


def faithfulness_test(
    image: torch.Tensor,            # (1, C, H, W)
    cam: torch.Tensor,              # (H, W) [0, 1]
    forward_fn: Callable[[torch.Tensor], torch.Tensor],  # image -> probs (1, n_classes)
    class_idx: int,
    n_steps: int = 10,
    baseline: str = "black",        # "black" | "blur"
) -> FaithfulnessResult:
    H, W = image.shape[2], image.shape[3]
    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=(H, W),
        mode="bilinear", align_corners=False
    ).squeeze()

    if baseline == "blur":
        base_image = F.avg_pool2d(image, kernel_size=11, stride=1, padding=5)
    else:
        base_image = torch.zeros_like(image)

    step_masks = _make_steps(cam_up, n_steps)
    deletion_curve, insertion_curve = [], []

    with torch.no_grad():
        for mask in step_masks:
            m = mask.unsqueeze(0).unsqueeze(0)
            # Deletion: start from full image, mask out top-k pixels.
            deleted = image * (1 - m) + base_image * m
            deletion_curve.append(float(forward_fn(deleted)[0, class_idx].item()))
            # Insertion: start from baseline, reveal top-k pixels.
            inserted = base_image * (1 - m) + image * m
            insertion_curve.append(float(forward_fn(inserted)[0, class_idx].item()))

    n = len(deletion_curve)
    del_auc = sum(deletion_curve) / n
    ins_auc = sum(insertion_curve) / n

    return FaithfulnessResult(
        deletion_auc=del_auc,
        insertion_auc=ins_auc,
        deletion_curve=deletion_curve,
        insertion_curve=insertion_curve,
        n_steps=n_steps,
    )