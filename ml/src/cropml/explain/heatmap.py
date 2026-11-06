"""Resize a CAM to the original image size, colorize, and encode as a
compact base64 PNG for the API's explanation.heatmap_png_base64 field."""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def cam_to_heatmap_pil(
    cam: torch.Tensor,          # (H, W) float [0,1]
    target_size: tuple[int, int],  # (W, H) output size
    overlay_image: Image.Image | None = None,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Resizes cam to target_size, applies a colormap, and optionally
    blends over the original image."""
    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(target_size[1], target_size[0]),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    cam_np = cam_up.detach().cpu().numpy()

    # Apply matplotlib-style colormap via a simple jet approximation
    # without importing matplotlib (keeps the dependency light).
    cam_uint8 = (cam_np * 255).astype(np.uint8)
    heatmap_rgb = _jet_colormap(cam_uint8)
    heatmap = Image.fromarray(heatmap_rgb)

    if overlay_image is not None:
        overlay_rgb = overlay_image.convert("RGB").resize(target_size)
        heatmap = Image.blend(overlay_rgb, heatmap, alpha=alpha)

    return heatmap


def _jet_colormap(gray: np.ndarray) -> np.ndarray:
    """Applies a jet-like colormap to a uint8 grayscale array.
    Returns (H, W, 3) uint8 RGB."""
    t = gray.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def cam_to_base64_png(
    cam: torch.Tensor,
    target_size: tuple[int, int],
    overlay_image: Image.Image | None = None,
    alpha: float = 0.5,
    max_dim: int = 224,
) -> str:
    """Returns a base64-encoded PNG string suitable for the API's
    explanation.heatmap_png_base64 field. Resizes to max_dim on the
    longest side to keep payload size small."""
    w, h = target_size
    scale = min(max_dim / max(w, h), 1.0)
    out_size = (max(1, int(w * scale)), max(1, int(h * scale)))

    heatmap = cam_to_heatmap_pil(cam, out_size, overlay_image, alpha)
    buf = io.BytesIO()
    heatmap.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")