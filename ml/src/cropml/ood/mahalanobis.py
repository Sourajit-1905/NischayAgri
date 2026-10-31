"""
Mahalanobis-distance OOD score, operating on BACKBONE EMBEDDINGS (not
logits). Lee et al. 2018 ("A Simple Unified Framework for Detecting
Out-of-Distribution Samples"). Per-class means, one shared (tied)
covariance across all classes - the original paper's formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class MahalanobisFitError(ValueError):
    """fit_mahalanobis was called with invalid or insufficient data."""


@dataclass(frozen=True)
class MahalanobisParams:
    class_means: torch.Tensor      # (n_classes, embedding_dim)
    precision: torch.Tensor        # (embedding_dim, embedding_dim), inverse of the tied covariance
    class_ids: tuple[int, ...]     # the class index each row of class_means corresponds to


def fit_mahalanobis(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    ridge: float = 1e-3,
) -> MahalanobisParams:
    """embeddings: (n, d), labels: (n,) integer class indices.
    Fits per-class means and one shared covariance (tied across classes),
    per Lee et al. 2018. ridge is added to the covariance diagonal before
    inverting, for numerical stability when d is large relative to n."""
    if embeddings.shape[0] != labels.shape[0]:
        raise MahalanobisFitError("embeddings and labels must have the same length")

    unique_classes = torch.unique(labels).tolist()
    if not unique_classes:
        raise MahalanobisFitError("no classes found in labels")

    embedding_dim = embeddings.shape[1]
    class_means = []
    centered_all = []

    for class_id in unique_classes:
        mask = labels == class_id
        class_embeddings = embeddings[mask]
        if class_embeddings.shape[0] < 2:
            raise MahalanobisFitError(
                f"class {class_id} has only {class_embeddings.shape[0]} sample(s) - "
                f"need at least 2 to contribute to the shared covariance estimate"
            )
        mean = class_embeddings.mean(dim=0)
        class_means.append(mean)
        centered_all.append(class_embeddings - mean)

    means_tensor = torch.stack(class_means, dim=0)
    centered = torch.cat(centered_all, dim=0)

    n = centered.shape[0]
    covariance = (centered.T @ centered) / max(n - 1, 1)
    covariance = covariance + ridge * torch.eye(embedding_dim)

    precision = torch.linalg.inv(covariance)

    return MahalanobisParams(
        class_means=means_tensor,
        precision=precision,
        class_ids=tuple(unique_classes),
    )


def mahalanobis_score(embeddings: torch.Tensor, params: MahalanobisParams) -> torch.Tensor:
    """embeddings: (batch, d) -> (batch,) OOD score.
    Higher score = more likely OOD (farther from every class centroid),
    matching msp_score/energy_score's sign convention."""
    # diff: (n_classes, batch, d)
    diff = embeddings.unsqueeze(0) - params.class_means.unsqueeze(1)
    # Mahalanobis distance per class: (n_classes, batch)
    distances = torch.einsum("cbd,de,cbe->cb", diff, params.precision, diff)
    # Minimum distance across classes = nearest class centroid -> (batch,)
    min_distances, _ = distances.min(dim=0)
    return min_distances