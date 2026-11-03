"""
CLI: train the severity head (CORAL ordinal regression) on cached
pseudo-severity labels, using an already-trained backbone's embeddings
(frozen by default).

Usage (from ml/):
    python scripts/train_severity.py
    python scripts/train_severity.py train.epochs=2 train.max_batches_per_epoch=5   # fast check
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cropml.models.severity_head import SeverityHeadConfig  # noqa: E402
from cropml.training.augmentation import AugmentationConfig  # noqa: E402
from cropml.training.severity_train_loop import (  # noqa: E402
    SeverityEpochMetrics,
    SeverityTrainConfig,
    train_severity,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _on_epoch_end(epoch: int, train_metrics: SeverityEpochMetrics, val_metrics: SeverityEpochMetrics) -> None:
    print(
        f"epoch {epoch}: "
        f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.3f} "
        f"train_mae={train_metrics.mean_abs_error:.3f} | "
        f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.3f} "
        f"val_mae={val_metrics.mean_abs_error:.3f}"
    )


@hydra.main(config_path=str(_CONFIG_DIR / "severity"), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    severity_head_config = SeverityHeadConfig(
        hidden_dim=cfg.ordinal_head.hidden_dim,
        dropout=cfg.ordinal_head.dropout,
        n_levels=cfg.ordinal_head.n_levels,
    )
    train_config = SeverityTrainConfig(
        epochs=cfg.train.epochs,
        seed=cfg.seed,
        device=cfg.train.device,
        batch_size=cfg.train.batch_size,
        learning_rate=cfg.train.learning_rate,
        num_workers=cfg.train.num_workers,
        freeze_backbone=cfg.train.freeze_backbone,
        max_batches_per_epoch=cfg.train.max_batches_per_epoch,
        checkpoint_dir=Path(cfg.checkpoint_dir),
    )
    run_name = f"{cfg.backbone_name}_severity_seed{cfg.seed}"

    print(f"=== training severity head on {cfg.backbone_name}, seed={cfg.seed}, epochs={cfg.train.epochs} ===")
    _, _, checkpoint_path = train_severity(
        backbone_name=cfg.backbone_name,
        dataset_name=cfg.dataset_name,
        manifest_path=cfg.manifest_path,
        cache_path=cfg.cache_path,
        severity_head_config=severity_head_config,
        train_config=train_config,
        aug_config=AugmentationConfig(),
        backbone_checkpoint_path=cfg.backbone_checkpoint_path,
        run_name=run_name,
        on_epoch_end=_on_epoch_end,
    )
    print(f"\ndone. checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()