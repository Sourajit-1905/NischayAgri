"""Cost-minimizing threshold search over DecisionConfig parameters,
producing cost-vs-coverage curves."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from cropml.decision.policy import Decision, DecisionConfig, DecisionResult, decide


@dataclass(frozen=True)
class CostConfig:
    wrong_treatment: float = 3.0
    missed_detection: float = 1.0
    unnecessary_abstain: float = 0.5
    false_ood_rejection: float = 0.8

    @classmethod
    def from_dict(cls, raw: dict) -> CostConfig:
        c = raw["costs"]
        return cls(
            wrong_treatment=float(c["wrong_treatment"]),
            missed_detection=float(c["missed_detection"]),
            unnecessary_abstain=float(c["unnecessary_abstain"]),
            false_ood_rejection=float(c["false_ood_rejection"]),
        )


@dataclass(frozen=True)
class ScoredSample:
    crop_confidence: float
    disease_confidence: float
    ood_score: float
    image_quality: float
    true_decision: Decision   # ground-truth: what the right decision was


def sample_cost(result: DecisionResult, truth: Decision, costs: CostConfig) -> float:
    """Cost of one decision given the ground-truth correct decision."""
    if result.action == truth:
        return 0.0
    if result.action == Decision.ACCEPT and truth != Decision.ACCEPT:
        return costs.wrong_treatment
    if result.action == Decision.REJECT_UNKNOWN and truth != Decision.REJECT_UNKNOWN:
        return costs.false_ood_rejection
    if result.action in (Decision.ABSTAIN_RETAKE, Decision.ABSTAIN_EXPERT):
        return costs.missed_detection if truth == Decision.ACCEPT else costs.unnecessary_abstain
    return costs.wrong_treatment  # fallback


def total_cost(
    samples: list[ScoredSample],
    config: DecisionConfig,
    costs: CostConfig,
) -> float:
    total = 0.0
    for s in samples:
        result = decide(s.crop_confidence, s.disease_confidence, s.ood_score, s.image_quality, config)
        total += sample_cost(result, s.true_decision, costs)
    return total / max(len(samples), 1)


@dataclass(frozen=True)
class TuningResult:
    best_config: DecisionConfig
    best_cost: float
    coverage: float   # fraction of samples that received ACCEPT under best_config


def tune_thresholds(
    samples: list[ScoredSample],
    costs: CostConfig,
    crop_thresholds: list[float] | None = None,
    disease_thresholds: list[float] | None = None,
    ood_thresholds: list[float] | None = None,
    quality_thresholds: list[float] | None = None,
) -> TuningResult:
    crop_thresholds = crop_thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
    disease_thresholds = disease_thresholds or [0.3, 0.4, 0.5, 0.6]
    ood_thresholds = ood_thresholds or [0.5, 0.6, 0.7, 0.8, 0.9]
    quality_thresholds = quality_thresholds or [0.3, 0.4, 0.5, 0.6]

    best_config = DecisionConfig()
    best_cost = float("inf")

    for c, d, o, q in product(crop_thresholds, disease_thresholds, ood_thresholds, quality_thresholds):
        cfg = DecisionConfig(crop_confidence_threshold=c, disease_confidence_threshold=d,
                             ood_score_threshold=o, image_quality_threshold=q)
        cost = total_cost(samples, cfg, costs)
        if cost < best_cost:
            best_cost = cost
            best_config = cfg

    n_accepted = sum(
        1 for s in samples
        if decide(s.crop_confidence, s.disease_confidence, s.ood_score, s.image_quality, best_config).action == Decision.ACCEPT
    )
    return TuningResult(best_config=best_config, best_cost=best_cost, coverage=n_accepted / max(len(samples), 1))


def cost_vs_coverage(
    samples: list[ScoredSample],
    costs: CostConfig,
    disease_thresholds: list[float] | None = None,
) -> list[tuple[float, float]]:
    """Sweeps disease_confidence_threshold (keeping other thresholds at
    defaults), returning (coverage, avg_cost) pairs for plotting."""
    thresholds = disease_thresholds or list(np.linspace(0.1, 0.9, 17))
    base = DecisionConfig()
    curve = []
    for t in thresholds:
        cfg = DecisionConfig(
            crop_confidence_threshold=base.crop_confidence_threshold,
            disease_confidence_threshold=t,
            ood_score_threshold=base.ood_score_threshold,
            image_quality_threshold=base.image_quality_threshold,
        )
        cost = total_cost(samples, cfg, costs)
        n_accepted = sum(
            1 for s in samples
            if decide(s.crop_confidence, s.disease_confidence, s.ood_score, s.image_quality, cfg).action == Decision.ACCEPT
        )
        curve.append((n_accepted / max(len(samples), 1), cost))
    return curve