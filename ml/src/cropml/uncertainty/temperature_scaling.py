"""Temperature scaling: post-hoc calibration via a scalar T per output
head, fit on a held-out validation set by minimizing NLL. Guo et al. 2017.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from cropml.models.backbone import Backbone
from cropml.models.hierarchical_head import HeadOutput, HierarchicalHead


@dataclass(frozen=True)
class CollectedLogits:
    crop_logits: torch.Tensor      # (n, n_crops)
    crop_labels: torch.Tensor      # (n,)
    disease_logits: torch.Tensor   # (n, n_classes)
    disease_labels: torch.Tensor   # (n,)


@torch.no_grad()
def collect_logits(
    backbone: Backbone, head: HierarchicalHead, loader: DataLoader, device: str
) -> CollectedLogits:
    """One pass over loader (typically the VAL split - never train or test)
    to gather raw logits and true labels for fitting/evaluating calibration."""
    backbone.eval()
    head.eval()

    all_crop_logits, all_crop_labels = [], []
    all_disease_logits, all_disease_labels = [], []

    for images, crop_idx, class_idx in loader:
        images = images.to(device)
        embeddings = backbone(images)
        out = head(embeddings)
        all_crop_logits.append(out.crop_logits.cpu())
        all_crop_labels.append(crop_idx)
        all_disease_logits.append(out.disease_logits.cpu())
        all_disease_labels.append(class_idx)

    return CollectedLogits(
        crop_logits=torch.cat(all_crop_logits),
        crop_labels=torch.cat(all_crop_labels),
        disease_logits=torch.cat(all_disease_logits),
        disease_labels=torch.cat(all_disease_labels),
    )


def fit_temperature(
    logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01
) -> float:
    """Fits a single scalar T minimizing NLL via L-BFGS - the standard
    approach for temperature scaling; converges in few iterations since
    there's only one parameter."""
    log_t = torch.zeros(1, requires_grad=True)  # optimize in log-space, T = exp(log_t), guarantees T > 0
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        t = log_t.exp()
        loss = criterion(logits / t, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


@dataclass(frozen=True)
class CalibrationResult:
    crop_temperature: float
    disease_temperature: float
    nll_before_crop: float
    nll_after_crop: float
    nll_before_disease: float
    nll_after_disease: float


def calibrate(backbone: Backbone, head: HierarchicalHead, val_loader: DataLoader, device: str) -> CalibrationResult:
    collected = collect_logits(backbone, head, val_loader, device)
    criterion = nn.CrossEntropyLoss()

    nll_before_crop = float(criterion(collected.crop_logits, collected.crop_labels).item())
    nll_before_disease = float(criterion(collected.disease_logits, collected.disease_labels).item())

    crop_t = fit_temperature(collected.crop_logits, collected.crop_labels)
    disease_t = fit_temperature(collected.disease_logits, collected.disease_labels)

    nll_after_crop = float(
        criterion(collected.crop_logits / crop_t, collected.crop_labels).item()
    )
    nll_after_disease = float(
        criterion(collected.disease_logits / disease_t, collected.disease_labels).item()
    )

    return CalibrationResult(
        crop_temperature=crop_t,
        disease_temperature=disease_t,
        nll_before_crop=nll_before_crop,
        nll_after_crop=nll_after_crop,
        nll_before_disease=nll_before_disease,
        nll_after_disease=nll_after_disease,
    )


class TemperatureScaledModel:
    """Wraps a trained backbone+head with fitted temperatures. This is
    what downstream code (metrics, OOD scorers, the eventual /predict
    decision logic) should call for calibrated probabilities - never the
    raw head output directly."""

    def __init__(
        self,
        backbone: Backbone,
        head: HierarchicalHead,
        crop_temperature: float,
        disease_temperature: float,
    ) -> None:
        self.backbone = backbone
        self.head = head
        self.crop_temperature = crop_temperature
        self.disease_temperature = disease_temperature

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> HeadOutput:
        self.backbone.eval()
        self.head.eval()
        embeddings = self.backbone(images)
        out = self.head(embeddings)
        calibrated_crop_probs = torch.softmax(out.crop_logits / self.crop_temperature, dim=-1)
        calibrated_disease_probs = torch.softmax(
            out.disease_logits / self.disease_temperature, dim=-1
        )
        return HeadOutput(
            crop_logits=out.crop_logits,
            disease_logits=out.disease_logits,
            crop_probs=calibrated_crop_probs,
            disease_probs=calibrated_disease_probs,
        )