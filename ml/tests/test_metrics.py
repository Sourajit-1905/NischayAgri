import torch

from cropml.evaluation.metrics import (
    accuracy,
    expected_calibration_error,
    fpr_at_95_tpr,
    macro_f1,
    ood_auroc,
    per_class_metrics,
    risk_coverage_curve,
)


def test_accuracy_and_macro_f1():
    preds = torch.tensor([0, 1, 1, 0, 1])
    labels = torch.tensor([0, 1, 0, 0, 1])
    assert abs(accuracy(preds, labels) - 0.8) < 1e-6
    assert 0 < macro_f1(preds, labels, n_classes=2) <= 1


def test_ece_badly_calibrated_exceeds_well_calibrated():
    good_conf = torch.tensor([0.9] * 10)
    good_correct = torch.tensor([1] * 9 + [0], dtype=torch.bool)
    bad_conf = torch.tensor([0.99] * 10)
    bad_correct = torch.tensor([1, 0] * 5, dtype=torch.bool)

    good_ece = expected_calibration_error(good_conf, good_correct, n_bins=10)
    bad_ece = expected_calibration_error(bad_conf, bad_correct, n_bins=10)
    assert bad_ece > good_ece


def test_ood_auroc_perfect_separation():
    id_scores = torch.tensor([0.1, 0.2, 0.15, 0.05])
    ood_scores = torch.tensor([0.9, 0.8, 0.95, 0.85])
    assert ood_auroc(id_scores, ood_scores) == 1.0
    assert fpr_at_95_tpr(id_scores, ood_scores) == 0.0


def test_risk_coverage_curve_aurc():
    conf = torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2])
    corr = torch.tensor([True, True, True, False, False])
    rc = risk_coverage_curve(conf, corr)
    assert abs(rc.aurc - 0.09) < 1e-6
    assert rc.risks[0] == 0.0  # most confident prediction is correct -> zero risk


def test_per_class_metrics_nan_below_threshold():
    preds = torch.tensor([0, 0, 1, 1, 1, 0])
    labels = torch.tensor([0, 0, 1, 1, 0, 0])
    conf = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.6, 0.9])
    results = per_class_metrics(preds, labels, conf, class_ids=["classA", "classB"], min_samples_for_ece=10)
    for r in results:
        assert r.ece != r.ece  # NaN check (NaN != NaN by IEEE 754)
        assert r.aurc != r.aurc

