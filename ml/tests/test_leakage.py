"""The phase's actual 'done when' gate: no group spans more than one split,
across a variety of scenarios that previously could have broken this."""

import pytest

from cropml.data.splitting import SplitRatios, assert_no_leakage, split_dataset


def _assert_truly_no_overlap(report) -> None:
    """Independent re-check, not reusing assert_no_leakage's own logic,
    so a bug in assert_no_leakage itself couldn't hide a real leak."""
    group_to_splits: dict[str, set[str]] = {}
    for a in report.assignments:
        group_to_splits.setdefault(a.group_id, set()).add(a.split)
    for gid, splits in group_to_splits.items():
        assert len(splits) == 1, f"group {gid} leaked across splits: {splits}"


@pytest.mark.parametrize("seed", [0, 1, 42, 123, 9999])
def test_no_leakage_across_many_seeds(seed, make_sample):
    samples = [make_sample(f"ds/h{i}.png", "tomato__healthy") for i in range(50)] + [
        make_sample(f"ds/b{i}.png", "tomato__early_blight") for i in range(50)
    ]
    report = split_dataset(samples, SplitRatios(0.7, 0.15, 0.15), seed=seed)
    assert_no_leakage(report)
    _assert_truly_no_overlap(report)


def test_no_leakage_with_many_dedup_clusters_of_varying_size(make_sample):
    samples = [make_sample(f"ds/img{i}.png", "tomato__healthy") for i in range(100)]
    # Clusters of varying size: 10, 5, 5, 5, ... , plus singletons
    dedup: dict[str, str] = {}
    cluster_sizes = [10, 5, 5, 5, 3, 3, 2, 2] + [1] * 20
    idx = 0
    for size in cluster_sizes:
        group_root = f"ds/img{idx}.png"
        for _ in range(size):
            dedup[f"ds/img{idx}.png"] = group_root
            idx += 1
    remaining = [make_sample(f"ds/img{i}.png", "tomato__healthy") for i in range(idx, 100)]
    samples = samples[:idx] + remaining

    report = split_dataset(samples, SplitRatios(0.7, 0.15, 0.15), seed=42, dedup_assignment=dedup)
    assert_no_leakage(report)
    _assert_truly_no_overlap(report)


def test_no_leakage_with_forced_test_and_dedup_combined(make_sample):
    samples = [make_sample(f"ds/img{i}.png", "tomato__healthy") for i in range(30)]
    dedup = {f"ds/img{i}.png": f"ds/img{i // 3 * 3}.png" for i in range(30)}  # groups of 3
    forced = frozenset(f"ds/img{i}.png" for i in range(0, 6))  # forces first two groups

    report = split_dataset(
        samples,
        SplitRatios(0.7, 0.15, 0.15),
        seed=7,
        dedup_assignment=dedup,
        forced_test_sample_ids=forced,
    )
    assert_no_leakage(report)
    _assert_truly_no_overlap(report)
    by_sample = report.as_dict_by_sample()
    for i in range(6):
        assert by_sample[f"ds/img{i}.png"] == "test"


def test_no_leakage_with_single_oversized_cluster_per_class(make_sample):
    """Regression coverage for the step-9 bug, at the leakage-test level
    specifically (not just the achieved-ratio check in test_splitting.py)."""
    samples = [make_sample(f"ds/img{i}.png", "tomato__healthy") for i in range(20)]
    dedup = {f"ds/img{i}.png": "ds/img0.png" for i in range(20)}  # ALL in one cluster
    report = split_dataset(samples, SplitRatios(0.7, 0.15, 0.15), seed=42, dedup_assignment=dedup)
    assert_no_leakage(report)
    _assert_truly_no_overlap(report)
    # With everything in one group, the whole group goes to a single split -
    # that's correct behavior (the group CANNOT be subdivided), not a leak.
    splits_used = {a.split for a in report.assignments}
    assert len(splits_used) == 1