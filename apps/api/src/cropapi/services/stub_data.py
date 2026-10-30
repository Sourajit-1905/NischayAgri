import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from cropapi.schemas.advice import AdviceResponse
from cropapi.schemas.common import AdviceMode, Decision
from cropapi.schemas.predict import PredictionResponse

# contracts/ lives at the repo root, three levels above this file:
# apps/api/src/cropapi/services/stub_data.py -> repo_root/contracts/examples
_CONTRACTS_EXAMPLES = (
    Path(__file__).resolve().parents[5] / "contracts" / "examples"
)

_PREDICT_FILES: dict[Decision, str] = {
    Decision.ACCEPT: "predict_accept.json",
    Decision.ABSTAIN_RETAKE: "predict_abstain_retake.json",
    Decision.ABSTAIN_EXPERT: "predict_abstain_expert.json",
    Decision.REJECT_UNKNOWN: "predict_reject_unknown.json",
}

_ADVICE_FILES: dict[AdviceMode, str] = {
    AdviceMode.FULL: "advice_full.json",
    AdviceMode.LIMITED: "advice_limited.json",
    AdviceMode.EXPERT_REFERRAL: "advice_expert_referral.json",
}

# Stubbed routing from a decision to the advice mode it should produce,
# mirroring the behaviour table in Phase 2.7's spec.
_DECISION_TO_ADVICE_MODE: dict[Decision, AdviceMode] = {
    Decision.ACCEPT: AdviceMode.FULL,
    Decision.ABSTAIN_RETAKE: AdviceMode.LIMITED,
    Decision.ABSTAIN_EXPERT: AdviceMode.EXPERT_REFERRAL,
    Decision.REJECT_UNKNOWN: AdviceMode.EXPERT_REFERRAL,
}




@lru_cache
def _load_json(filename: str) -> dict[str, Any]:
    path = _CONTRACTS_EXAMPLES / filename
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


def get_stub_prediction(decision: Decision) -> PredictionResponse:
    data = _load_json(_PREDICT_FILES[decision])
    return PredictionResponse.model_validate(data)


def get_stub_advice_for_decision(decision: Decision) -> AdviceResponse:
    mode = _DECISION_TO_ADVICE_MODE[decision]
    data = _load_json(_ADVICE_FILES[mode])
    return AdviceResponse.model_validate(data)


def cycle_stub_decision(counter: int) -> Decision:
    """Deterministically rotate through decisions so repeated stub calls
    are exercisable end-to-end without real inference."""
    decisions = list(_PREDICT_FILES.keys())
    return decisions[counter % len(decisions)]