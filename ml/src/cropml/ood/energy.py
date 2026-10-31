"""
Energy-based OOD score. Liu et al. 2020 ("Energy-based Out-of-
distribution Detection"). Operates on raw LOGITS, not softmax
probabilities - uses magnitude information softmax normalization discards.
"""

from __future__ import annotations

import torch


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """logits: (batch, n_classes) -> (batch,) OOD score.
    Higher score = more likely OOD (matches msp_score's sign convention).

    Free energy E(x) = -T * logsumexp(logits / T). Confident, in-distribution
    predictions (one large logit) produce very NEGATIVE energy; uncertain/
    OOD predictions (small, similar logits) produce energy closer to zero
    (i.e. a LARGER value) - so E(x) itself is already in the "higher = more
    OOD" direction and needs no further sign flip.
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)



