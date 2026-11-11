"""
CLI: precompute lesion-area-ratio pseudo-severity for every image in a
dataset, caching results to disk (configs/severity/default.yaml's
cache_path). Avoids recomputing this heuristic every training epoch.

Healthy images get severity "none" directly from their class label (no
pixel analysis needed). Diseased images are bucketed low/moderate/high
via the lesion-ratio heuristic, calibrated against this project's real
PlantVillage diseased-class distribution (see pseudo_severity.py).

Usage (from ml/):
    python scripts/compute_pseudo_severity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cropml.data.loaders import get_loader  # noqa: E402
from cropml.data.pseudo_severity import (  # noqa: E402
    SeverityConfig,
    bucket_known_diseased,
    estimate_severity_from_path,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _color_path_to_segmented(color_path: Path) -> Path:
    """Swaps the 'color' path segment for 'segmented' AND appends
    '_final_masked' before the extension - confirmed against this
    download's actual naming (color: '<id>___<label>.JPG', segmented:
    '<id>___<label>_final_masked.jpg'; case differs too, color/ is .JPG,
    segmented/ is lowercase .jpg). Searches the segmented class folder
    case-insensitively rather than assuming an exact extension match,
    since relying on the color file's exact-case extension would be
    fragile against any future download with different casing."""
    parts = list(color_path.parts)
    if "color" not in parts:
        raise ValueError(f"'{color_path}' has no 'color' path segment to swap")
    idx = parts.index("color")
    parts[idx] = "segmented"
    segmented_dir = Path(*parts[:-1])
    stem = color_path.stem

    candidate = segmented_dir / f"{stem}_final_masked.jpg"
    if candidate.exists():
        return candidate

    if segmented_dir.is_dir():
        target_prefix = f"{stem}_final_masked".lower()
        for f in segmented_dir.iterdir():
            if f.is_file() and f.stem.lower() == target_prefix:
                return f

    return candidate


@hydra.main(config_path=str(_CONFIG_DIR / "severity"), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    severity_config = SeverityConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))

    print(f"loading samples for dataset '{cfg.dataset_name}'...")
    samples = get_loader(cfg.dataset_name).load()
    print(f"loaded {len(samples)} samples")

    cache: dict[str, dict] = {}
    missing_segmented = 0
    bucket_counts: dict[str, int] = {}

    for i, sample in enumerate(samples):
        if i % 5000 == 0:
            print(f"  processed {i}/{len(samples)}...")

        is_healthy = sample.class_id.endswith("__healthy")

        if is_healthy:
            cache[sample.sample_id] = {"lesion_ratio": 0.0, "bucket": "none", "leaf_pixel_count": None}
            bucket_counts["none"] = bucket_counts.get("none", 0) + 1
            continue

        try:
            segmented_path = _color_path_to_segmented(sample.path)
        except ValueError as e:
            print(f"  WARNING: {e}")
            missing_segmented += 1
            continue

        if not segmented_path.exists():
            missing_segmented += 1
            continue

        estimate = estimate_severity_from_path(segmented_path, severity_config)
        bucket = bucket_known_diseased(estimate.lesion_ratio, severity_config)
        cache[sample.sample_id] = {
            "lesion_ratio": estimate.lesion_ratio,
            "bucket": bucket.value,
            "leaf_pixel_count": estimate.leaf_pixel_count,
        }
        bucket_counts[bucket.value] = bucket_counts.get(bucket.value, 0) + 1

    print(f"\nprocessed {len(cache)} images ({missing_segmented} missing segmented counterparts, skipped)")
    print(f"bucket distribution: {bucket_counts}")

    cache_path = Path(cfg.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f)
    print(f"\ncache written to: {cache_path}")


if __name__ == "__main__":
    main()


    