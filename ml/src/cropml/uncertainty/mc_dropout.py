"""
Monte Carlo dropout: N stochastic forward passes with dropout kept
active, used to estimate predictive (epistemic) uncertainty.
Gal & Ghahramani 2016; entropy/MI decomposition per Kwon et al. 2020.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cropml.models.backbone import Backbone
from cropml.models.hierarchical_head import HierarchicalHead


@dataclass(frozen=True)
class MCDropoutResult:
    mean_crop_probs: torch.Tensor        # (batch, n_crops) - averaged over samples
    mean_disease_probs: torch.Tensor     # (batch, n_classes)
    crop_predictive_entropy: torch.Tensor     # (batch,) - entropy of the mean prediction
    crop_mutual_information: torch.Tensor     # (batch,) - epistemic component (disagreement)
    disease_predictive_entropy: torch.Tensor  # (batch,)
    disease_mutual_information: torch.Tensor  # (batch,)
    n_samples: int


def _entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy along the last dim. probs: (..., n_classes) -> (...,)"""
    return -(probs * (probs + eps).log()).sum(dim=-1)


@torch.no_grad()
def mc_dropout_predict(
    backbone: Backbone,
    head: HierarchicalHead,
    images: torch.Tensor,
    n_samples: int = 20,
) -> MCDropoutResult:
    """Runs n_samples stochastic forward passes with dropout active, and
    decomposes uncertainty into predictive entropy and mutual information
    (the epistemic/disagreement component)."""
    backbone.eval()
    head.eval()
    backbone.enable_mc_dropout()
    try:
        crop_probs_samples = []
        disease_probs_samples = []
        for _ in range(n_samples):
            embeddings = backbone(images)
            out = head(embeddings)
            crop_probs_samples.append(out.crop_probs)
            disease_probs_samples.append(out.disease_probs)
    finally:
        # Always restore normal eval-mode dropout, even if something above
        # raised - otherwise the model is left in a stuck stochastic state
        # for any code that uses it afterward.
        backbone.disable_mc_dropout()

    crop_stack = torch.stack(crop_probs_samples, dim=0)        # (n_samples, batch, n_crops)
    disease_stack = torch.stack(disease_probs_samples, dim=0)  # (n_samples, batch, n_classes)

    mean_crop_probs = crop_stack.mean(dim=0)
    mean_disease_probs = disease_stack.mean(dim=0)

    # Predictive entropy: entropy of the MEAN prediction (total uncertainty).
    crop_predictive_entropy = _entropy(mean_crop_probs)
    disease_predictive_entropy = _entropy(mean_disease_probs)

    # Expected entropy: average entropy OF EACH individual sample
    # (aleatoric component - uncertainty inherent to each single pass).
    crop_expected_entropy = _entropy(crop_stack).mean(dim=0)
    disease_expected_entropy = _entropy(disease_stack).mean(dim=0)

    # Mutual information = predictive entropy - expected entropy.
    # This isolates epistemic uncertainty: how much the samples DISAGREE
    # with each other, beyond what any single sample's own entropy explains.
    crop_mi = crop_predictive_entropy - crop_expected_entropy
    disease_mi = disease_predictive_entropy - disease_expected_entropy

    return MCDropoutResult(
        mean_crop_probs=mean_crop_probs,
        mean_disease_probs=mean_disease_probs,
        crop_predictive_entropy=crop_predictive_entropy,
        crop_mutual_information=crop_mi,
        disease_predictive_entropy=disease_predictive_entropy,
        disease_mutual_information=disease_mi,
        n_samples=n_samples,
    )