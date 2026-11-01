"""CLI: the "one command" for Phase 2.3's done-when criterion. Trains
across the (backbone x seed) grid once, evaluates each trained model
under every method (softmax / temperature_scaling / mc_dropout /
deep_ensemble), and produces the final results table (mean +/- std over
seeds, per backbone x method x target_domain).

Usage (from ml/):
    python scripts/run_experiment_table.py
    python scripts/run_experiment_table.py max_eval_samples=50 train.epochs=2 train.max_batches_per_epoch=5  # fast sanity check
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cropml.data.label_space import load_label_space  # noqa: E402
from cropml.evaluation.metrics import accuracy, expected_calibration_error, macro_f1, risk_coverage_curve  # noqa: E402
from cropml.evaluation.results_table import (  # noqa: E402
    RunResult,
    aggregate_over_seeds,
    build_display_table,
    build_results_table,
    to_markdown_table,
)
from cropml.models import get_backbone  # noqa: E402
from cropml.models.hierarchical_head import HierarchicalHead  # noqa: E402
from cropml.training.augmentation import AugmentationConfig, build_eval_transform  # noqa: E402
from cropml.training.dataset import CropDiseaseDataset  # noqa: E402
from cropml.training.train_loop import TrainConfig, run_training  # noqa: E402
from cropml.uncertainty.deep_ensemble import load_ensemble_members, ensemble_predict  # noqa: E402
from cropml.uncertainty.mc_dropout import mc_dropout_predict  # noqa: E402
from cropml.uncertainty.temperature_scaling import calibrate, TemperatureScaledModel  # noqa: E402

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# Only plantvillage_test is wired up for now. Extend this once PlantDoc/
# FieldPlant manifests exist (via Phase 2.2's build_splits.py run against
# those datasets): add entries like "plantdoc": ("plantdoc", "test").
_DOMAIN_TO_DATASET_SPLIT: dict[str, tuple[str, str]] = {
    "plantvillage_test": ("plantvillage", "test"),
}


def _train_or_load(backbone_name: str, seed: int, cfg: DictConfig):
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"{backbone_name}_seed{seed}.pt"

    if checkpoint_path.exists():
        print(f"  [{backbone_name} seed={seed}] checkpoint exists, loading (skip training)")
        backbone = get_backbone(backbone_name)
        head = HierarchicalHead(embedding_dim=backbone.embedding_dim)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        backbone.load_state_dict(checkpoint["backbone_state_dict"])
        head.load_state_dict(checkpoint["head_state_dict"])
        backbone.eval()
        head.eval()
        return backbone, head, checkpoint_path

    print(f"  [{backbone_name} seed={seed}] training...")
    train_config = TrainConfig(
        epochs=cfg.train.epochs,
        seed=seed,
        device=cfg.train.device,
        num_workers=cfg.train.num_workers,
        use_class_weights=cfg.train.use_class_weights,
        lambda_crop=cfg.train.lambda_crop,
        lambda_disease=cfg.train.lambda_disease,
        max_batches_per_epoch=cfg.train.max_batches_per_epoch,
        checkpoint_dir=checkpoint_dir,
    )
    aug_config = AugmentationConfig(
        use_color_augmentation=cfg.augmentation.use_color_augmentation,
        use_gan_data=cfg.augmentation.use_gan_data,
    )
    backbone, head, checkpoint_path = run_training(
        backbone_name=backbone_name,
        dataset_name=cfg.dataset_name,
        manifest_path=cfg.manifest_path,
        train_config=train_config,
        aug_config=aug_config,
        run_name=f"{backbone_name}_seed{seed}",
    )
    backbone.eval()
    head.eval()
    return backbone, head, checkpoint_path


def _build_eval_loader(
    domain: str, head: HierarchicalHead, input_size: int, batch_size: int, max_samples: int | None
) -> DataLoader:
    dataset_name, split = _DOMAIN_TO_DATASET_SPLIT[domain]
    eval_transform = build_eval_transform(input_size)
    ds = CropDiseaseDataset(dataset_name, f"data/manifests/{dataset_name}.json", split, head, transform=eval_transform)
    if max_samples is not None and max_samples < len(ds):
        # Dataset order follows the loader's path-sorted order (Phase 2.2), so
        # taking the FIRST max_samples would likely select one single class
        # (whichever sorts first alphabetically) rather than a representative
        # sample across all classes - this was causing accuracy=0.000 on
        # capped sanity runs. Use a fixed-seed random sample instead.
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(ds), generator=generator)[:max_samples].tolist()
        ds = Subset(ds, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _metrics_from_probs(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    preds = probs.argmax(dim=-1)
    confidences = probs.max(dim=-1).values
    correct = preds == labels
    rc = risk_coverage_curve(confidences, correct)
    return {
        "accuracy": accuracy(preds, labels),
        "macro_f1": macro_f1(preds, labels, n_classes=probs.shape[-1]),
        "ece": expected_calibration_error(confidences, correct),
        "aurc": rc.aurc,
    }


@torch.no_grad()
def _evaluate_softmax(backbone, head, loader: DataLoader) -> dict[str, float]:
    backbone.eval()
    head.eval()
    all_probs, all_labels = [], []
    for images, _, class_idx in loader:
        out = head(backbone(images))
        all_probs.append(out.disease_probs)
        all_labels.append(class_idx)
    return _metrics_from_probs(torch.cat(all_probs), torch.cat(all_labels))


def _evaluate_temperature_scaling(backbone, head, val_loader: DataLoader, test_loader: DataLoader) -> dict[str, float]:
    result = calibrate(backbone, head, val_loader, device="cpu")
    scaled = TemperatureScaledModel(backbone, head, result.crop_temperature, result.disease_temperature)
    all_probs, all_labels = [], []
    for images, _, class_idx in test_loader:
        out = scaled.predict(images)
        all_probs.append(out.disease_probs)
        all_labels.append(class_idx)
    return _metrics_from_probs(torch.cat(all_probs), torch.cat(all_labels))


def _evaluate_mc_dropout(backbone, head, loader: DataLoader, n_samples: int) -> dict[str, float]:
    all_probs, all_labels = [], []
    for images, _, class_idx in loader:
        result = mc_dropout_predict(backbone, head, images, n_samples=n_samples)
        all_probs.append(result.mean_disease_probs)
        all_labels.append(class_idx)
    return _metrics_from_probs(torch.cat(all_probs), torch.cat(all_labels))


def _evaluate_deep_ensemble(backbone_name: str, checkpoint_paths: list[Path], n_members: int, loader: DataLoader) -> dict[str, float]:
    members = load_ensemble_members(backbone_name, checkpoint_paths[:n_members])
    all_probs, all_labels = [], []
    for images, _, class_idx in loader:
        result = ensemble_predict(members, images)
        all_probs.append(result.mean_disease_probs)
        all_labels.append(class_idx)
    return _metrics_from_probs(torch.cat(all_probs), torch.cat(all_labels))


@hydra.main(config_path=str(_CONFIG_DIR / "experiment"), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    label_space = load_label_space()  # noqa: F841 - validated importable/loadable early
    results: list[RunResult] = []

    for backbone_name in cfg.backbones:
        print(f"\n=== {backbone_name} ===")
        checkpoints_by_seed: dict[int, Path] = {}
        backbones_by_seed = {}
        heads_by_seed = {}

        for seed in cfg.seeds:
            backbone, head, checkpoint_path = _train_or_load(backbone_name, seed, cfg)
            checkpoints_by_seed[seed] = checkpoint_path
            backbones_by_seed[seed] = backbone
            heads_by_seed[seed] = head

        for domain in cfg.target_domains:
            print(f"  evaluating on {domain}...")
            sample_head = heads_by_seed[cfg.seeds[0]]
            sample_backbone = backbones_by_seed[cfg.seeds[0]]

            for seed in cfg.seeds:
                backbone = backbones_by_seed[seed]
                head = heads_by_seed[seed]
                test_loader = _build_eval_loader(
                    domain, head, backbone.config.input_size, backbone.config.batch_size, cfg.max_eval_samples
                )

                if "softmax" in cfg.methods:
                    metrics = _evaluate_softmax(backbone, head, test_loader)
                    results.append(RunResult(backbone_name, "softmax", domain, seed, metrics))
                    print(f"    seed={seed} softmax: acc={metrics['accuracy']:.3f} ece={metrics['ece']:.3f}")

                if "temperature_scaling" in cfg.methods:
                    val_ds_name, _ = _DOMAIN_TO_DATASET_SPLIT[domain]
                    val_eval_transform = build_eval_transform(backbone.config.input_size)
                    val_ds = CropDiseaseDataset(
                        val_ds_name, f"data/manifests/{val_ds_name}.json", "val", head, transform=val_eval_transform
                    )
                    if cfg.max_eval_samples is not None and cfg.max_eval_samples < len(val_ds):
                        generator = torch.Generator().manual_seed(0)
                        indices = torch.randperm(len(val_ds), generator=generator)[: cfg.max_eval_samples].tolist()
                        val_ds = Subset(val_ds, indices)
                    val_loader = DataLoader(val_ds, batch_size=backbone.config.batch_size, shuffle=False)

                    metrics = _evaluate_temperature_scaling(backbone, head, val_loader, test_loader)
                    results.append(RunResult(backbone_name, "temperature_scaling", domain, seed, metrics))
                    print(f"    seed={seed} temperature_scaling: acc={metrics['accuracy']:.3f} ece={metrics['ece']:.3f}")

                if "mc_dropout" in cfg.methods:
                    metrics = _evaluate_mc_dropout(backbone, head, test_loader, cfg.mc_dropout.n_samples)
                    results.append(RunResult(backbone_name, "mc_dropout", domain, seed, metrics))
                    print(f"    seed={seed} mc_dropout: acc={metrics['accuracy']:.3f} ece={metrics['ece']:.3f}")

            if "deep_ensemble" in cfg.methods:
                checkpoint_list = [checkpoints_by_seed[s] for s in cfg.seeds]
                ensemble_loader = _build_eval_loader(
                    domain, sample_head, sample_backbone.config.input_size,
                    sample_backbone.config.batch_size, cfg.max_eval_samples,
                )
                metrics = _evaluate_deep_ensemble(backbone_name, checkpoint_list, cfg.deep_ensemble.n_members, ensemble_loader)
                # seed=-1 sentinel: this is ONE result combining multiple seeds,
                # not a per-seed result - results_table.py's "(n=1)" display
                # correctly represents this as not-averaged-over-seeds.
                results.append(RunResult(backbone_name, "deep_ensemble", domain, -1, metrics))
                print(f"    deep_ensemble (n_members={cfg.deep_ensemble.n_members}): acc={metrics['accuracy']:.3f} ece={metrics['ece']:.3f}")

    raw_df = build_results_table(results)
    agg_df = aggregate_over_seeds(raw_df)
    display_df = build_display_table(agg_df, metrics=["accuracy", "macro_f1", "ece", "aurc"])

    print("\n\n=== RESULTS TABLE ===\n")
    print(display_df.to_string(index=False))

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(results_dir / "raw_results.csv", index=False)
    (results_dir / "results_table.md").write_text(to_markdown_table(display_df), encoding="utf-8")
    print(f"\nsaved: {results_dir / 'raw_results.csv'}, {results_dir / 'results_table.md'}")


if __name__ == "__main__":
    main()



    