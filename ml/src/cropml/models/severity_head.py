"""
Ordinal severity head: CORAL (Cao, Mirjalili & Raschka 2020). A single
shared score per sample, compared against K-1 learned thresholds via
weight-sharing, which respects severity's natural ordering
(none < low < moderate < high) - unlike treating it as K unordered classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SeverityHeadConfig:
    hidden_dim: int
    dropout: float
    n_levels: int

    @classmethod
    def from_dict(cls, raw: dict) -> SeverityHeadConfig:
        return cls(
            hidden_dim=int(raw["hidden_dim"]),
            dropout=float(raw["dropout"]),
            n_levels=int(raw["n_levels"]),
        )


@dataclass(frozen=True)
class SeverityOutput:
    cumulative_logits: torch.Tensor   # (batch, n_levels - 1)
    cumulative_probs: torch.Tensor    # (batch, n_levels - 1), P(level > k)


class SeverityHead(nn.Module):
    def __init__(self, embedding_dim: int, config: SeverityHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.n_levels = config.n_levels

        self.feature = nn.Sequential(
            nn.Linear(embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        # Single shared weight vector -> one score per sample. This is
        # CORAL's defining trick: EVERY threshold compares the SAME score,
        # just with a different learned bias - not K-1 independent heads.
        self.shared_score = nn.Linear(config.hidden_dim, 1, bias=False)
        self.thresholds = nn.Parameter(torch.zeros(config.n_levels - 1))

    def forward(self, embeddings: torch.Tensor) -> SeverityOutput:
        features = self.feature(embeddings)
        base_score = self.shared_score(features)  # (batch, 1)
        cumulative_logits = base_score + self.thresholds.unsqueeze(0)  # (batch, n_levels-1)
        cumulative_probs = torch.sigmoid(cumulative_logits)
        return SeverityOutput(cumulative_logits=cumulative_logits, cumulative_probs=cumulative_probs)


def coral_loss(cumulative_logits: torch.Tensor, true_levels: torch.Tensor, n_levels: int) -> torch.Tensor:
    """true_levels: (batch,) integer level indices in [0, n_levels-1].
    Converts each true level into its K-1 binary "is level > k" targets,
    then sums per-threshold binary cross-entropy - the CORAL training
    objective."""
    batch_size = true_levels.shape[0]
    thresholds = torch.arange(n_levels - 1, device=true_levels.device).unsqueeze(0)  # (1, n_levels-1)
    targets = (true_levels.unsqueeze(1) > thresholds).float()  # (batch, n_levels-1)
    return F.binary_cross_entropy_with_logits(cumulative_logits, targets, reduction="mean") * (
        batch_size / batch_size  # no-op, kept for clarity that this is a mean over all (batch x threshold) terms
    )


def decode_level(cumulative_probs: torch.Tensor) -> torch.Tensor:
    """Standard CORAL decode: predicted level = how many thresholds the
    score exceeds (P(level > k) > 0.5). Returns (batch,) integer levels."""
    return (cumulative_probs > 0.5).sum(dim=-1)


def level_probabilities(cumulative_probs: torch.Tensor) -> torch.Tensor:
    """Converts K-1 cumulative probabilities into K per-level probabilities
    via consecutive differences, for entropy-based uncertainty.

    NOTE: cumulative_probs are not mathematically guaranteed monotonically
    decreasing for any single prediction (CORAL's rank-consistency is a
    training-time/aggregate property, not a per-sample guarantee), so raw
    differences can occasionally be slightly negative. Clamped at 0 and
    renormalized here rather than letting that leak into downstream use."""
    batch_size, n_thresholds = cumulative_probs.shape
    n_levels = n_thresholds + 1

    p_level_0 = 1.0 - cumulative_probs[:, 0:1]
    p_level_last = cumulative_probs[:, -1:]
    if n_thresholds > 1:
        p_middle = cumulative_probs[:, :-1] - cumulative_probs[:, 1:]
        probs = torch.cat([p_level_0, p_middle, p_level_last], dim=1)
    else:
        probs = torch.cat([p_level_0, p_level_last], dim=1)

    probs = probs.clamp(min=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    assert probs.shape == (batch_size, n_levels)
    return probs


def severity_entropy(level_probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy over the per-level probability distribution -
    uncertainty on the severity level itself, as called for in the plan."""
    return -(level_probs * (level_probs + eps).log()).sum(dim=-1)