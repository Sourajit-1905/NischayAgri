"""Grad-CAM: gradient-weighted class activation mapping.
Selvaraj et al. 2017. Hook-based, works with any backbone that has
a spatial conv feature map (MobileNetV3, EfficientNet-B0).
ViT requires attention rollout instead - flagged but not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class GradCAMError(ValueError):
    pass


@dataclass(frozen=True)
class CAMResult:
    cam: torch.Tensor   # (H, W) float, values in [0, 1], same spatial size as target layer
    class_idx: int
    score: float        # logit value for the target class


class GradCAM:
    """Hook-based Grad-CAM. target_layer should be the last spatial
    conv/feature layer of the backbone (before global pooling).

    Usage:
        with GradCAM(backbone, target_layer) as cam:
            result = cam.compute(image, class_idx)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list = []

    def __enter__(self) -> GradCAM:
        self._handles.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self._handles.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )
        return self

    def __exit__(self, *_) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _save_activation(self, _module, _input, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output: tuple) -> None:
        self._gradients = grad_output[0].detach()
  
    def compute(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
        class_idx: int | None = None,
    ) -> CAMResult:
        if self._activations is None:
            raise GradCAMError(
                "No activations recorded. Ensure the forward pass traversed "
                "target_layer inside the 'with GradCAM(...)' context."
            )
        if class_idx is None:
            class_idx = int(logits.argmax(dim=-1).item())

        # Trigger backward to populate gradients via the hook.
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)

        if self._gradients is None:
            raise GradCAMError(
                "No gradients recorded after backward. Check that target_layer "
                "is on the gradient path and that inputs have requires_grad."
            )

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return CAMResult(
            cam=cam.squeeze(),
            class_idx=class_idx,
            score=float(logits[0, class_idx].item()),
        )


def get_target_layer(backbone_name: str, timm_model: nn.Module) -> nn.Module:
    """Returns the last spatial feature layer for supported backbones.
    This is architecture-specific and must be updated if new backbones
    are added."""
    if "mobilenet" in backbone_name:
        return timm_model.blocks[-1]
    if "efficientnet" in backbone_name:
        return timm_model.blocks[-1]
    if "vit" in backbone_name:
        raise GradCAMError(
            "ViT lacks spatial conv feature maps - Grad-CAM is not applicable. "
            "Use attention rollout for ViT explainability (not implemented here)."
        )
    raise GradCAMError(f"Unknown backbone '{backbone_name}' - add it to get_target_layer().")