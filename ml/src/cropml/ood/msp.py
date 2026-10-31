"""Maximum Softmax Probability (MSP) OOD score. Hendrycks & Gimpel 2017.

Operates on already-computed probabilities (ideally temperature-calibrated,
since MSP is sensitive to miscalibration) rather than raw model output -
keeps this decoupled from any specific model-wrapping class.
"""

from __future__ import annotations

import torch


def msp_score(probs: torch.Tensor) -> torch.Tensor:
    """probs: (batch, n_classes) -> (batch,) OOD score in [0, 1].
    Higher score = more likely OOD (less confident prediction)."""
    max_prob, _ = probs.max(dim=-1)
    return 1.0 - max_prob