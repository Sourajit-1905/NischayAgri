"""
CLI: build leakage-free split manifests for one or more datasets.

Usage (from ml/):
    python scripts/build_splits.py
    python scripts/build_splits.py datasets=[plantvillage]
    python scripts/build_splits.py seed=7 ratios.train=0.8 ratios.val=0.1 ratios.test=0.1
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Allow running as `python scripts/build_splits.py` without an editable
# install, by adding src/ to the path when cropml isn't already importable.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cropml.data.dedup import cluster_near_duplicates, compute_phashes  # noqa: E402
from cropml.data.loaders import default_config_dir, get_loader  # noqa: E402
from cropml.data.loaders.base import DatasetSanityWarning, Sample  # noqa: E402
from cropml.data.manifest import save_manifest  # noqa: E402
from cropml.data.splitting import SplitRatios, assert_no_leakage, split_dataset  # noqa: E402

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _forced_test_ids(samples: list[Sample], dataset_name: str, cfg: DictConfig) -> frozenset[str]:
    if not cfg.respect_official_test_split.get(dataset_name, False):
        return frozenset()
    return frozenset(s.sample_id for s in samples if s.split == "test")


def _build_one(dataset_name: str, cfg: DictConfig) -> None:
    print(f"\n=== {dataset_name} ===")

    loader = get_loader(dataset_name, config_dir=default_config_dir())
    samples = loader.load()
    print(f"loaded {len(samples)} samples, {len({s.class_id for s in samples})} classes")

    hashed = compute_phashes(samples, hash_size=cfg.dedup.hash_size)
    dedup_assignment = cluster_near_duplicates(hashed, max_distance=cfg.dedup.max_distance)
    n_clusters = len(set(dedup_assignment.values()))
    print(f"deduped into {n_clusters} groups ({len(samples) - n_clusters} near-duplicates found)")

    forced = _forced_test_ids(samples, dataset_name, cfg)
    if forced:
        print(f"{len(forced)} samples forced into 'test' (official split)")

    ratios = SplitRatios(cfg.ratios.train, cfg.ratios.val, cfg.ratios.test)
    report = split_dataset(
        samples,
        ratios=ratios,
        seed=cfg.seed,
        dedup_assignment=dedup_assignment,
        forced_test_sample_ids=forced,
    )
    assert_no_leakage(report)
    print("leakage check: passed")

    for class_id, achieved in sorted(report.achieved_ratios_by_class.items()):
        print(f"  {class_id:45s} {achieved}")

    manifest_path = Path(cfg.manifest_dir) / f"{dataset_name}.json"
    config_path = default_config_dir() / f"{dataset_name}.yaml"
    save_manifest(
        report,
        manifest_path,
        dataset_names=[dataset_name],
        config_paths=[config_path],
    )
    print(f"manifest written to {manifest_path}")


@hydra.main(config_path=str(_CONFIG_DIR / "splits"), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    import warnings

    for dataset_name in cfg.datasets:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DatasetSanityWarning)
            _build_one(dataset_name, cfg)
            for w in caught:
                print(f"  WARNING: {w.message}")


if __name__ == "__main__":
    main()