"""
Leakage-free, group-aware dataset splitting.

The split unit is always a GROUP (a dedup cluster, or a real metadata key
when a dataset provides one) - never an individual sample. This guarantees
no near-duplicate or same-source image can appear in more than one split.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from cropml.data.loaders.base import Sample

SplitName = str  # "train" | "val" | "test", but kept open for other names


class SplitConfigError(ValueError):
    """
    The requested split configuration is invalid.
    """


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0, atol=1e-6):
            raise SplitConfigError(f"ratios must sum to 1.0, got {total}")
        if min(self.train, self.val, self.test) < 0:
            raise SplitConfigError("ratios must be non-negative")

    def as_dict(self) -> dict[SplitName, float]:
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass(frozen=True)
class SplitAssignment:
    sample_id: str
    split: SplitName
    group_id: str


@dataclass(frozen=True)
class SplitReport:
    assignments: tuple[SplitAssignment, ...]
    achieved_ratios_by_class: dict[str, dict[SplitName, float]]
    target_ratios: dict[SplitName, float]
    seed: int
    forced_test_sample_ids: frozenset[str] = field(default_factory=frozenset)

    def as_dict_by_sample(self) -> dict[str, SplitName]:
        return {a.sample_id: a.split for a in self.assignments}


def _majority_class(samples: list[Sample]) -> str:
    """The class with the most samples in this group. Ties broken by the
    lexicographically smallest class_id, for determinism."""
    counts: dict[str, int] = defaultdict(int)
    for s in samples:
        counts[s.class_id] += 1
    max_count = max(counts.values())
    tied = sorted(c for c, n in counts.items() if n == max_count)
    return tied[0]



def _majority_class_simple(samples: list[Sample]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for s in samples:
        counts[s.class_id] += 1
    max_count = max(counts.values())
    tied = sorted(c for c, n in counts.items() if n == max_count)
    return tied[0]


def build_groups(
    samples: list[Sample],
    dedup_assignment: dict[str, str] | None,
) -> dict[str, list[Sample]]:
    """
    Map group_id -> its samples.
    If dedup_assignment is given, it is the group id for every sample it
    covers. Samples not covered by dedup_assignment (e.g. a dataset that
    hasn't been through dedup.py yet) each become their own singleton group.
    """
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        group_id = (dedup_assignment or {}).get(s.sample_id, s.sample_id)
        groups[group_id].append(s)
    return dict(groups)


def split_dataset(
    samples: list[Sample],
    ratios: SplitRatios,
    seed: int,
    dedup_assignment: dict[str, str] | None = None,
    forced_test_sample_ids: frozenset[str] = frozenset(),
) -> SplitReport:
    """
    Assign every sample to train/val/test at the group level.

    forced_test_sample_ids: samples that MUST land in "test" regardless of
    ratios (e.g. PlantDoc's official test split, per target_eval_only).
    Any group containing a forced sample is entirely forced to "test", so a
    group can never straddle forced and non-forced splits.
    """
    if not samples:
        raise SplitConfigError("no samples to split")

    groups = build_groups(samples, dedup_assignment)

    forced_group_ids = {
        gid
        for gid, group_samples in groups.items()
        if any(s.sample_id in forced_test_sample_ids for s in group_samples)
    }
    free_group_ids = sorted(set(groups) - forced_group_ids)

    # Stratify free groups by their majority class, then shuffle within each
    # class bucket deterministically before slicing by ratio.
    by_class: dict[str, list[str]] = defaultdict(list)
    for gid in free_group_ids:
        by_class[_majority_class(groups[gid])].append(gid)

    rng = np.random.default_rng(seed)
    split_of_group: dict[str, SplitName] = {gid: "test" for gid in forced_group_ids}

    for class_id in sorted(by_class):
        gids = by_class[class_id]
        order = rng.permutation(len(gids))
        gids_shuffled = [gids[i] for i in order]

        sizes = [len(groups[g]) for g in gids_shuffled]
        total = sum(sizes)
        cum = np.cumsum(sizes)
        train_cut = np.searchsorted(cum, ratios.train * total, side="left")
        val_cut = np.searchsorted(cum, (ratios.train + ratios.val) * total, side="left")

        for i, gid in enumerate(gids_shuffled):
            if i < train_cut:
                split_of_group[gid] = "train"
            elif i < val_cut:
                split_of_group[gid] = "val"
            else:
                split_of_group[gid] = "test"

    assignments = [
        SplitAssignment(sample_id=s.sample_id, split=split_of_group[gid], group_id=gid)
        for gid, group_samples in groups.items()
        for s in group_samples
    ]
    assignments.sort(key=lambda a: a.sample_id)

    achieved = _achieved_ratios_by_class(samples, {a.sample_id: a.split for a in assignments})

    return SplitReport(
        assignments=tuple(assignments),
        achieved_ratios_by_class=achieved,
        target_ratios=ratios.as_dict(),
        seed=seed,
        forced_test_sample_ids=forced_test_sample_ids,
    )


def _achieved_ratios_by_class(
    samples: list[Sample], split_of_sample: dict[str, SplitName]
) -> dict[str, dict[SplitName, float]]:
    counts: dict[str, dict[SplitName, int]] = defaultdict(lambda: defaultdict(int))
    for s in samples:
        counts[s.class_id][split_of_sample[s.sample_id]] += 1
    result: dict[str, dict[SplitName, float]] = {}
    for class_id, split_counts in counts.items():
        total = sum(split_counts.values())
        result[class_id] = {
            split: round(n / total, 4) for split, n in sorted(split_counts.items())
        }
    return result


def assert_no_leakage(report: SplitReport) -> None:
    """
    Raise AssertionError if any group appears in more than one split.

    This is the function the leakage test (test_leakage.py) calls directly;
    it is also the gate CI should run before trusting any split manifest.
    """
    group_splits: dict[str, set[SplitName]] = defaultdict(set)
    for a in report.assignments:
        group_splits[a.group_id].add(a.split)
    leaking = {gid: splits for gid, splits in group_splits.items() if len(splits) > 1}
    if leaking:
        raise AssertionError(f"{len(leaking)} group(s) span more than one split: {leaking}")