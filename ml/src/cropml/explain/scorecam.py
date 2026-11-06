"""Score-CAM: gradient-free CAM via activation masking.
Wang et al. 2020. Slower than Grad-CAM (one forward pass per channel)
but doesn't require gradients, so works for non-differentiable pipelines
and tends to produce less noisy maps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class CAMResult:
    cam: torch.Tensor
    class_idx: int
    score: float


def score_cam(
    image: torch.Tensor,                        # (1, C, H, W)
    target_layer: nn.Module,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],  # image -> logits (1, n_classes)
    class_idx: int | None = None,
    max_channels: int = 64,                     # cap for CPU feasibility
) -> CAMResult:
    """forward_fn must be differentiable but Score-CAM itself never calls
    backward - it only calls forward_fn once per activation channel.
    max_channels caps the number of channels used (top-max by L2 norm)
    to keep CPU runtime manageable."""
    activations: list[torch.Tensor] = []

    def _hook(_module, _input, output: torch.Tensor) -> None:
        activations.append(output.detach())

    handle = target_layer.register_forward_hook(_hook)
    with torch.no_grad():
        baseline_logits = forward_fn(image)
    handle.remove()

    if not activations:
        raise ValueError("No activations recorded - check target_layer is on the forward path.")

    act = activations[0]  # (1, n_channels, h, w)
    n_channels = act.shape[1]
    H, W = image.shape[2], image.shape[3]

    if class_idx is None:
        class_idx = int(baseline_logits.argmax(dim=-1).item())

    # Select top channels by L2 norm to stay within max_channels budget.
    norms = act[0].norm(dim=(1, 2))
    top_k = min(max_channels, n_channels)
    top_indices = norms.topk(top_k).indices

    scores = torch.zeros(top_k)
    with torch.no_grad():
        for i, ch in enumerate(top_indices):
            # Upscale activation channel to image size, normalize to [0,1].
            mask = act[0, ch].unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
            mask_min, mask_max = mask.min(), mask.max()
            if (mask_max - mask_min) > 1e-8:
                mask = (mask - mask_min) / (mask_max - mask_min)

            masked_image = image * mask
            logits = forward_fn(masked_image)
            scores[i] = logits[0, class_idx]

    # Softmax-normalize scores -> weights, then weighted sum of upscaled activations.
    weights = torch.softmax(scores, dim=0)
    cam = torch.zeros(1, 1, H, W)
    for i, ch in enumerate(top_indices):
        mask = act[0, ch].unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
        cam += weights[i] * mask

    cam = F.relu(cam)
    cam_min, cam_max = cam.min(), cam.max()
    if (cam_max - cam_min) > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)

    return CAMResult(
        cam=cam.squeeze(),
        class_idx=class_idx,
        score=float(baseline_logits[0, class_idx].item()),
    )