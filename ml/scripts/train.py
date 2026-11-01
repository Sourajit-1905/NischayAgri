"""
CLI: train one model (one backbone, one seed). Training is
method-agnostic - "method" (softmax/temperature_scaling/mc_dropout/
deep_ensemble) is applied post-hoc to a trained checkpoint by evaluation
code, not something this script branches on. For deep ensembles, run
this script once per seed; run_experiment_table.py (step 17) orchestrates
that.

Usage (from ml/):
    python scripts/train.py
    python scripts/train.py backbone=efficientnet_b0 seed=1
    python scripts/train.py train.epochs=2 train.max_batches_per_epoch=5   # fast check
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cropml.training.augmentation import AugmentationConfig  # noqa: E402
from cropml.training.train_loop import EpochMetrics, TrainConfig, run_training  # noqa: E402

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _on_epoch_end(epoch: int, train_metrics: EpochMetrics, val_metrics: EpochMetrics) -> None:
    print(
        f"epoch {epoch}: "
        f"train_loss={train_metrics.loss:.4f} "
        f"train_crop_acc={train_metrics.crop_accuracy:.3f} "
        f"train_disease_acc={train_metrics.disease_accuracy:.3f} | "
        f"val_loss={val_metrics.loss:.4f} "
        f"val_crop_acc={val_metrics.crop_accuracy:.3f} "
        f"val_disease_acc={val_metrics.disease_accuracy:.3f}"
    )


@hydra.main(config_path=str(_CONFIG_DIR / "experiment"), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    train_config = TrainConfig(
        epochs=cfg.train.epochs,
        seed=cfg.seed,
        device=cfg.train.device,
        num_workers=cfg.train.num_workers,
        use_class_weights=cfg.train.use_class_weights,
        lambda_crop=cfg.train.lambda_crop,
        lambda_disease=cfg.train.lambda_disease,
        max_batches_per_epoch=cfg.train.max_batches_per_epoch,
        checkpoint_dir=Path(cfg.checkpoint_dir),
    )
    aug_config = AugmentationConfig(
        use_color_augmentation=cfg.augmentation.use_color_augmentation,
        use_gan_data=cfg.augmentation.use_gan_data,
    )
    run_name = f"{cfg.backbone}_seed{cfg.seed}"

    print(f"=== training {cfg.backbone}, seed={cfg.seed}, epochs={cfg.train.epochs} ===")
    _, _, checkpoint_path = run_training(
        backbone_name=cfg.backbone,
        dataset_name=cfg.dataset_name,
        manifest_path=cfg.manifest_path,
        train_config=train_config,
        aug_config=aug_config,
        run_name=run_name,
        on_epoch_end=_on_epoch_end,
    )
    print(f"\ndone. checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()


