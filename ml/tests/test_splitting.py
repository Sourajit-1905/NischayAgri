import pytest

from cropml.data.splitting import (
    SplitConfigError,
    SplitRatios,
    assert_no_leakage,
    split_dataset,
)


def _samples(make_sample, n_per_class: int = 20) -> list:
    return [
        make_sample(f"ds/healthy_{i}.png", "tomato__healthy") for i in range(n_per_class)
    ] + [
        make_sample(f"ds/blight_{i}.png", "tomato__early_blight") for i in range(n_per_class)
    ]


def test_ratios_must_sum_to_one():
    with pytest.raises(SplitConfigError):
        SplitRatios(0.5, 0.3, 0.3)


def test_split_is_deterministic_given_same_seed(make_sample):
    samples = _samples(make_sample)
    r1 = split_dataset(samples, SplitRatios(0.7, 0.15, 0.15), seed=42)
    r2 = split_dataset(samples, SplitRatios(0.7, 0.15, 0.15), seed=42)
    assert r1.as_dict_by_sample() == r2.as_dict_by_sample()


def test_different_seeds_can_produce_different_splits(make_sample):
    samples = _samples(make_sample)
    r1 = split_dataset(samples, SplitRatios(0.7, 0.15, 0.15), seed=1)
    r2 = split_dataset(samples, SplitRatios(0.7, 0.15, 0.15), seed=2)
    assert r1.as_dict_by_sample() != r2.as_dict_by_sample()


def test_grouped_samples_always_land_in_the_same_split(make_sample):
    samples = _samples(make_sample)
    dedup = {
        "ds/healthy_0.png": "ds/healthy_0.png",
        "ds/healthy_1.png": "ds/healthy_0.png",
        "ds/healthy_2.png": "ds/healthy_0.png",
    }
    report = split_dataset(samples, SplitRatios(0.7, 0.15, 0.15), seed=42, dedup_assignment=dedup)
    by_sample = report.as_dict_by_sample()
    splits = {by_sample[k] for k in dedup}
    assert len(splits) == 1


def test_no_leakage_passes_on_valid_split(make_sample):
    samples = _samples(make_sample)
    report = split_dataset(samples, SplitRatios(0.7, 0.15, 0.15), seed=42)
    assert_no_leakage(report)  # must not raise


def test_dominant_cluster_does_not_strand_a_split_at_zero(make_sample):
    """Regression test for the step-9 bug: one oversized group in a class
    must not cause train or val to end up with nothing."""
    samples = [make_sample(f"ds/img{i}.png", "tomato__healthy") for i in range(9)] + [
        make_sample("ds/unique.png", "tomato__healthy")
    ]
    dedup = {f"ds/img{i}.png": "ds/img0.png" for i in range(9)}  # one big cluster of 9
    report = split_dataset(samples, SplitRatios(0.7, 0.15, 0.15), seed=42, dedup_assignment=dedup)
    achieved = report.achieved_ratios_by_class["tomato__healthy"]
    assert achieved.get("train", 0) > 0  # train must not be stranded at zero


def test_forced_test_samples_stay_in_test(make_sample):
    samples = _samples(make_sample)
    forced = frozenset(["ds/healthy_0.png"])
    report = split_dataset(
        samples, SplitRatios(0.7, 0.15, 0.15), seed=42, forced_test_sample_ids=forced
    )
    assert report.as_dict_by_sample()["ds/healthy_0.png"] == "test"


def test_forced_group_is_entirely_forced_to_test(make_sample):
    """A group containing even one forced sample must be ENTIRELY in test,
    never straddling forced/non-forced across the same group."""
    samples = _samples(make_sample)
    dedup = {"ds/healthy_0.png": "ds/healthy_0.png", "ds/healthy_1.png": "ds/healthy_0.png"}
    forced = frozenset(["ds/healthy_0.png"])
    report = split_dataset(
        samples,
        SplitRatios(0.7, 0.15, 0.15),
        seed=42,
        dedup_assignment=dedup,
        forced_test_sample_ids=forced,
    )
    by_sample = report.as_dict_by_sample()
    assert by_sample["ds/healthy_0.png"] == "test"
    assert by_sample["ds/healthy_1.png"] == "test"


def test_empty_sample_list_rejected():
    with pytest.raises(SplitConfigError):
        split_dataset([], SplitRatios(0.7, 0.15, 0.15), seed=42)