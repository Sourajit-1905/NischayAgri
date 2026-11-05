import pytest
from cropml.decision.policy import Decision, DecisionConfig, decide

def _cfg(**kwargs):
    return DecisionConfig(**kwargs)

def test_accept():
    r = decide(0.9, 0.8, 0.1, 0.9, _cfg())
    assert r.action == Decision.ACCEPT

def test_abstain_retake_bad_quality():
    r = decide(0.9, 0.8, 0.1, 0.3, _cfg())
    assert r.action == Decision.ABSTAIN_RETAKE

def test_reject_unknown_ood():
    r = decide(0.9, 0.8, 0.9, 0.9, _cfg())
    assert r.action == Decision.REJECT_UNKNOWN

def test_abstain_expert_low_crop():
    r = decide(0.3, 0.8, 0.1, 0.9, _cfg())
    assert r.action == Decision.ABSTAIN_EXPERT

def test_abstain_expert_low_disease():
    r = decide(0.9, 0.2, 0.1, 0.9, _cfg())
    assert r.action == Decision.ABSTAIN_EXPERT

def test_quality_checked_before_ood():
    # bad quality AND ood - quality rule fires first
    r = decide(0.9, 0.8, 0.9, 0.3, _cfg())
    assert r.action == Decision.ABSTAIN_RETAKE

def test_all_branches_covered():
    cfg = _cfg()
    decisions = {
        decide(0.9, 0.8, 0.1, 0.9, cfg).action,
        decide(0.9, 0.8, 0.1, 0.3, cfg).action,
        decide(0.9, 0.8, 0.9, 0.9, cfg).action,
        decide(0.3, 0.8, 0.1, 0.9, cfg).action,
    }
    assert decisions == {Decision.ACCEPT, Decision.ABSTAIN_RETAKE, Decision.REJECT_UNKNOWN, Decision.ABSTAIN_EXPERT}