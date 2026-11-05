import random
from cropml.decision.policy import Decision
from cropml.decision.threshold_tuning import (
    CostConfig, ScoredSample, sample_cost, total_cost,
    tune_thresholds, cost_vs_coverage, DecisionResult,
)

def _samples(n=50, seed=0):
    random.seed(seed)
    return [ScoredSample(
        crop_confidence=random.uniform(0.4, 1.0),
        disease_confidence=random.uniform(0.3, 1.0),
        ood_score=random.uniform(0.0, 0.4),
        image_quality=random.uniform(0.6, 1.0),
        true_decision=Decision.ACCEPT,
    ) for _ in range(n)]

def test_sample_cost_zero_when_correct():
    costs = CostConfig()
    assert sample_cost(DecisionResult(Decision.ACCEPT, ""), Decision.ACCEPT, costs) == 0.0

def test_sample_cost_nonzero_when_wrong():
    costs = CostConfig()
    assert sample_cost(DecisionResult(Decision.ACCEPT, ""), Decision.REJECT_UNKNOWN, costs) > 0

def test_tune_thresholds_returns_valid_config():
    result = tune_thresholds(_samples(), CostConfig(),
        crop_thresholds=[0.4, 0.5], disease_thresholds=[0.3, 0.4],
        ood_thresholds=[0.7], quality_thresholds=[0.5])
    assert 0 <= result.coverage <= 1
    assert result.best_cost >= 0

def test_cost_vs_coverage_monotone_coverage():
    curve = cost_vs_coverage(_samples(), CostConfig(),
        disease_thresholds=[0.2, 0.4, 0.6, 0.8])
    coverages = [c for c, _ in curve]
    # higher threshold = less coverage (monotonically decreasing)
    assert coverages == sorted(coverages, reverse=True)