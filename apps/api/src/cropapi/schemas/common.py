from enum import StrEnum


class Decision(StrEnum):
    ACCEPT = "ACCEPT"
    ABSTAIN_RETAKE = "ABSTAIN_RETAKE"
    ABSTAIN_EXPERT = "ABSTAIN_EXPERT"
    REJECT_UNKNOWN = "REJECT_UNKNOWN"


class SeverityLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class AdviceMode(StrEnum):
    FULL = "FULL"
    LIMITED = "LIMITED"
    EXPERT_REFERRAL = "EXPERT_REFERRAL"