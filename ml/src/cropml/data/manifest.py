"""
Persisting and loading split manifests.

A manifest is the durable, shareable record of a split: which sample_id
went to which split, plus enough metadata to tell whether it's gone stale
relative to the configs/label space that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cropml.data.splitting import SplitAssignment, SplitReport

MANIFEST_SCHEMA_VERSION = 1


class ManifestError(ValueError):
    """
    The manifest file is malformed or doesn't match its schema.
    """


def _hash_config_files(paths: list[Path]) -> str:
    """
    A combined hash of the given files' contents, so a manifest can
    detect if any config it was built from has since changed."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda p: p.name):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def save_manifest(
    report: SplitReport,
    path: Path | str,
    dataset_names: list[str],
    config_paths: list[Path],
) -> None:
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": report.seed,
        "target_ratios": report.target_ratios,
        "dataset_names": sorted(dataset_names),
        "config_hash": _hash_config_files(config_paths),
        "forced_test_sample_ids": sorted(report.forced_test_sample_ids),
        "achieved_ratios_by_class": report.achieved_ratios_by_class,
        "assignments": [asdict(a) for a in report.assignments],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: Path | str) -> SplitReport:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"{path}: schema_version {raw.get('schema_version')!r} is not "
            f"the supported {MANIFEST_SCHEMA_VERSION}"
        )
    required = (
        "seed", "target_ratios", "achieved_ratios_by_class", "assignments",
    )  # fmt: skip
    missing = [k for k in required if k not in raw]
    if missing:
        raise ManifestError(f"{path}: missing keys {missing}")

    assignments = tuple(
        SplitAssignment(sample_id=a["sample_id"], split=a["split"], group_id=a["group_id"])
        for a in raw["assignments"]
    )
    return SplitReport(
        assignments=assignments,
        achieved_ratios_by_class=raw["achieved_ratios_by_class"],
        target_ratios=raw["target_ratios"],
        seed=raw["seed"],
        forced_test_sample_ids=frozenset(raw.get("forced_test_sample_ids", [])),
    )


def manifest_is_stale(path: Path | str, config_paths: list[Path]) -> bool:
    """
    True if any of config_paths has changed since the manifest was built,
    or the manifest doesn't exist.
    """
    
    p = Path(path)
    if not p.exists():
        return True
    raw = json.loads(p.read_text(encoding="utf-8"))
    current_hash = _hash_config_files(config_paths)
    return bool(raw.get("config_hash") != current_hash)




