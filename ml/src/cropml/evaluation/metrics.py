"""

Evaluation metrics: accuracy, macro-F1, ECE, OOD AUROC/FPR95,
risk-coverage/AURC, and per-class breakdowns for long-tail classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    return float((predictions == labels).float().mean().item())


def macro_f1(predictions: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    return float(
        f1_score(
            labels.cpu().numpy(),
            predictions.cpu().numpy(),
            labels=list(range(n_classes)),
            average="macro",
            zero_division=0,
        )
    )


def expected_calibration_error(
    confidences: torch.Tensor, correct: torch.Tensor, n_bins: int = 15
) -> float:
    """confidences: (n,) max predicted probability per sample.
    correct: (n,) bool/0-1, whether the prediction was actually right.
    ECE = sum over bins of (bin_weight * |bin_accuracy - bin_avg_confidence|).
    """
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same shape")

    confidences_np = confidences.cpu().numpy()
    correct_np = correct.float().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    n = len(confidences_np)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences_np > lo) & (confidences_np <= hi) if i > 0 else (
            (confidences_np >= lo) & (confidences_np <= hi)
        )
        bin_count = in_bin.sum()
        if bin_count == 0:
            continue
        bin_accuracy = correct_np[in_bin].mean()
        bin_confidence = confidences_np[in_bin].mean()
        ece += (bin_count / n) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def ood_auroc(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    """id_scores/ood_scores: OOD scores (higher = more OOD) for in-
    distribution and out-of-distribution samples respectively. AUROC
    treats OOD as the positive class - 1.0 is perfect separation."""
    scores = torch.cat([id_scores, ood_scores]).cpu().numpy()
    is_ood = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    return float(roc_auc_score(is_ood, scores))


def fpr_at_95_tpr(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    """The false-positive rate (fraction of ID samples incorrectly
    flagged as OOD) when the threshold is set so 95% of true OOD samples
    are correctly detected. Lower is better."""
    sorted_ood = torch.sort(ood_scores).values
    # threshold at the 5th percentile of OOD scores -> 95% of OOD scores
    # are above it (correctly flagged as OOD).
    idx = max(int(0.05 * len(sorted_ood)) - 1, 0)
    threshold = sorted_ood[idx]
    false_positives = (id_scores >= threshold).float().mean()
    return float(false_positives.item())


@dataclass(frozen=True)
class RiskCoverageResult:
    coverages: torch.Tensor   # (n,) fraction of samples accepted, descending confidence
    risks: torch.Tensor       # (n,) error rate among accepted samples, at each coverage level
    aurc: float               # area under the risk-coverage curve - lower is better


def risk_coverage_curve(confidences: torch.Tensor, correct: torch.Tensor) -> RiskCoverageResult:
    """Sorts by confidence descending; at each point, 'coverage' is what
    fraction of all samples are accepted (the most confident k), and
    'risk' is the error rate among just those accepted samples. AURC is
    the area under this curve - a model that concentrates its errors in
    the low-confidence tail (good selective prediction) has low AURC."""
    order = torch.argsort(confidences, descending=True)
    sorted_correct = correct[order].float()

    n = len(sorted_correct)
    cumulative_errors = torch.cumsum(1.0 - sorted_correct, dim=0)
    coverages = torch.arange(1, n + 1, dtype=torch.float32) / n
    risks = cumulative_errors / torch.arange(1, n + 1, dtype=torch.float32)

    # Trapezoidal integration of risk over coverage, as AURC.
    aurc = float(torch.trapz(risks, coverages).item())

    return RiskCoverageResult(coverages=coverages, risks=risks, aurc=aurc)


@dataclass(frozen=True)
class ClassMetrics:
    class_id: str
    n_samples: int
    accuracy: float
    ece: float
    aurc: float


def per_class_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    class_ids: list[str],
    min_samples_for_ece: int = 10,
) -> list[ClassMetrics]:
    """Per-class accuracy/ECE/AURC - specifically for examining long-tail
    classes (e.g. potato__healthy, apple__cedar_apple_rust - confirmed
    real long-tail classes in this project's PlantVillage data), where
    aggregate metrics could hide poor calibration on rare classes.
    Classes with fewer than min_samples_for_ece samples get ECE/AURC set
    to NaN rather than a misleadingly precise number from too few points.
    """
    results = []
    for class_idx, class_id in enumerate(class_ids):
        mask = labels == class_idx
        n = int(mask.sum().item())
        if n == 0:
            continue

        class_preds = predictions[mask]
        class_labels = labels[mask]
        class_conf = confidences[mask]
        class_correct = class_preds == class_labels

        acc = accuracy(class_preds, class_labels)
        if n >= min_samples_for_ece:
            ece = expected_calibration_error(class_conf, class_correct, n_bins=min(10, n // 2))
            rc = risk_coverage_curve(class_conf, class_correct)
            aurc = rc.aurc
        else:
            ece = float("nan")
            aurc = float("nan")

        results.append(
            ClassMetrics(class_id=class_id, n_samples=n, accuracy=acc, ece=ece, aurc=aurc)
        )
    return results




