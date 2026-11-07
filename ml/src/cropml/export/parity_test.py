"""
Numeric parity check between PyTorch and ONNX Runtime outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cropml.export.onnx_export import ExportConfig


@dataclass(frozen=True)
class ParityResult:
    crop_logits_max_diff: float
    disease_logits_max_diff: float
    passed: bool
    atol: float


def check_parity(
    onnx_path: Path | str,
    backbone,
    head,
    config: ExportConfig | None = None,
    atol: float = 1e-3,
    n_samples: int = 4,
) -> ParityResult:
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("onnxruntime is required for parity testing") from e

    config = config or ExportConfig()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    torch.manual_seed(0)
    images = torch.randn(n_samples, 3, config.input_size, config.input_size)

    backbone.eval(); head.eval()
    with torch.no_grad():
        embeddings = backbone(images)
        pt_out = head(embeddings)
        pt_crop = pt_out.crop_logits.numpy()
        pt_disease = pt_out.disease_logits.numpy()

    ort_outputs = session.run(
        ["crop_logits", "disease_logits"],
        {"images": images.numpy()},
    )
    ort_crop, ort_disease = ort_outputs[0], ort_outputs[1]

    crop_diff = float(np.abs(pt_crop - ort_crop).max())
    disease_diff = float(np.abs(pt_disease - ort_disease).max())
    passed = crop_diff <= atol and disease_diff <= atol

    return ParityResult(
        crop_logits_max_diff=crop_diff,
        disease_logits_max_diff=disease_diff,
        passed=passed,
        atol=atol,
    )