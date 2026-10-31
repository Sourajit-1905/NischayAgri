"""Thin wrapper around MLflow, local-file-tracking only (no server, no
cost). train_loop.py goes through this interface rather than calling
mlflow directly, keeping tracking swappable and testable.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

_DEFAULT_TRACKING_DIR = Path(__file__).resolve().parents[3] / "mlruns"


def _flatten_config(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flattens a dataclass (or plain dict) into a flat {key: value} dict
    suitable for mlflow.log_params. Nested dataclasses get dotted keys,
    e.g. backbone.learning_rate."""
    result: dict[str, Any] = {}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        items = dataclasses.asdict(obj).items()
    elif isinstance(obj, dict):
        items = obj.items()
    else:
        return {prefix.rstrip("."): obj}

    for key, value in items:
        full_key = f"{prefix}{key}"
        if dataclasses.is_dataclass(value) or isinstance(value, dict):
            result.update(_flatten_config(value, prefix=f"{full_key}."))
        else:
            result[full_key] = value
    return result


class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str = "crop_disease_uncertainty",
        tracking_dir: Path | None = None,
    ) -> None:
        self.tracking_dir = tracking_dir or _DEFAULT_TRACKING_DIR
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

        # MLflow's plain folder-based file store is in maintenance mode and
        # being phased out (as of the mlflow version installed here) - use
        # the SQLite backend instead. Still fully local and free, no server
        # process required; just stored as a .db file instead of raw folders.
        db_path = self.tracking_dir / "mlflow.db"
        artifact_dir = self.tracking_dir / "mlartifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_dir.as_uri())
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(
        self, run_name: str, configs: dict[str, Any] | None = None
    ) -> Iterator[None]:
        """Usage:
            with tracker.start_run("mobilenet_v3_seed0", {"backbone": cfg}):
                tracker.log_metric("train_loss", 0.42, step=0)
        Any exception inside the block still ends the run (as FAILED
        rather than leaving it open), via mlflow's own context manager.
        """
        with mlflow.start_run(run_name=run_name):
            if configs:
                for name, cfg in configs.items():
                    params = _flatten_config(cfg, prefix=f"{name}.")
                    # mlflow param values must be strings; cast everything.
                    mlflow.log_params({k: str(v) for k, v in params.items()})
            yield

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: Path | str) -> None:
        mlflow.log_artifact(str(local_path))

    def set_tags(self, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)

    def current_run_id(self) -> str | None:
        active = mlflow.active_run()
        return active.info.run_id if active else None