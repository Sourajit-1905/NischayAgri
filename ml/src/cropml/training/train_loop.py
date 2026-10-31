"""Ties backbone + head + dataset + augmentation + tracking together into
an actual trainable pipeline. First point where a model learns from data.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cropml.data.label_space import load_label_space
from cropml.models import get_backbone
from cropml.models.backbone import Backbone
from cropml.models.hierarchical_head import HierarchicalHead
from cropml.training.augmentation import (
    AugmentationConfig,
    build_eval_transform,
    build_train_transform,
)
from cropml.training.dataset import CropDiseaseDataset
from cropml.training.tracking import ExperimentTracker

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "checkpoints"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 0
    use_class_weights: bool = True
    lambda_crop: float = 1.0
    lambda_disease: float = 1.0
    max_batches_per_epoch: int | None = None  # cap, for fast CPU verification
    log_every_n_steps: int = 10
    checkpoint_dir: Path = field(default_factory=lambda: _DEFAULT_CHECKPOINT_DIR)


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    crop_accuracy: float
    disease_accuracy: float


def _compute_class_weights(
    train_ds: CropDiseaseDataset, head: HierarchicalHead
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse-frequency weights, for the crop head and disease head
    respectively. Addresses the real imbalance found in Phase 2.3 step 7
    (e.g. potato__healthy=152 vs orange__citrus_greening=5507)."""
    disease_counts_by_class = train_ds.class_counts()

    disease_counts = torch.ones(len(head.classes))
    for class_id, count in disease_counts_by_class.items():
        disease_counts[head.class_index(class_id)] = count
    disease_weights = disease_counts.sum() / (len(disease_counts) * disease_counts)

    crop_counts = torch.zeros(len(head.crops))
    for class_id, count in disease_counts_by_class.items():
        crop = head.label_space.get(class_id).crop
        crop_counts[head.crop_index(crop)] += count
    crop_counts = crop_counts.clamp(min=1)
    crop_weights = crop_counts.sum() / (len(crop_counts) * crop_counts)

    return crop_weights, disease_weights


def build_dataloaders(
    dataset_name: str,
    manifest_path: Path | str,
    head: HierarchicalHead,
    backbone_input_size: int,
    aug_config: AugmentationConfig,
    train_config: TrainConfig,
) -> tuple[DataLoader, DataLoader, CropDiseaseDataset]:
    train_transform = build_train_transform(aug_config, backbone_input_size)
    eval_transform = build_eval_transform(backbone_input_size)

    train_ds = CropDiseaseDataset(
        dataset_name, manifest_path, "train", head, transform=train_transform
    )
    val_ds = CropDiseaseDataset(
        dataset_name, manifest_path, "val", head, transform=eval_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=head.label_space and _batch_size_for(head),  # placeholder, replaced below
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=_batch_size_for(head),
        shuffle=False,
        num_workers=train_config.num_workers,
    )
    return train_loader, val_loader, train_ds


def _batch_size_for(head: HierarchicalHead) -> int:  # noqa: ARG001
    # Placeholder resolved properly in run_training, which has access to
    # the actual BackboneConfig.batch_size. Kept here only so
    # build_dataloaders has a sane standalone default for direct testing.
    return 16


def _run_epoch(
    backbone: Backbone,
    head: HierarchicalHead,
    loader: DataLoader,
    device: str,
    crop_weights: torch.Tensor,
    disease_weights: torch.Tensor,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer | None,
) -> EpochMetrics:
    """optimizer=None runs in eval mode (no gradient updates)."""
    is_train = optimizer is not None
    backbone.train(is_train)
    head.train(is_train)

    crop_criterion = nn.CrossEntropyLoss(weight=crop_weights.to(device))
    disease_criterion = nn.CrossEntropyLoss(weight=disease_weights.to(device))

    total_loss = 0.0
    total_crop_correct = 0
    total_disease_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (images, crop_idx, class_idx) in enumerate(loader):
            if config.max_batches_per_epoch is not None and batch_idx >= config.max_batches_per_epoch:
                break

            images = images.to(device)
            crop_idx = crop_idx.to(device)
            class_idx = class_idx.to(device)

            embeddings = backbone(images)
            out = head(embeddings)

            crop_loss = crop_criterion(out.crop_logits, crop_idx)
            disease_loss = disease_criterion(out.disease_logits, class_idx)
            loss = config.lambda_crop * crop_loss + config.lambda_disease * disease_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_crop_correct += (out.crop_logits.argmax(dim=-1) == crop_idx).sum().item()
            total_disease_correct += (out.disease_logits.argmax(dim=-1) == class_idx).sum().item()
            total_samples += batch_size

    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        crop_accuracy=total_crop_correct / max(total_samples, 1),
        disease_accuracy=total_disease_correct / max(total_samples, 1),
    )


def run_training(
    backbone_name: str,
    dataset_name: str,
    manifest_path: Path | str,
    train_config: TrainConfig,
    aug_config: AugmentationConfig | None = None,
    run_name: str | None = None,
    on_epoch_end: Callable[[int, EpochMetrics, EpochMetrics], None] | None = None,
) -> tuple[Backbone, HierarchicalHead, Path]:
    """Trains one backbone end-to-end. Returns (backbone, head, checkpoint_path)."""
    set_seed(train_config.seed)
    aug_config = aug_config or AugmentationConfig()

    backbone = get_backbone(backbone_name)
    label_space = load_label_space()
    head = HierarchicalHead(embedding_dim=backbone.embedding_dim, label_space=label_space)
    backbone.to(train_config.device)
    head.to(train_config.device)

    train_transform = build_train_transform(aug_config, backbone.config.input_size)
    eval_transform = build_eval_transform(backbone.config.input_size)
    train_ds = CropDiseaseDataset(
        dataset_name, manifest_path, "train", head, transform=train_transform
    )
    val_ds = CropDiseaseDataset(
        dataset_name, manifest_path, "val", head, transform=eval_transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=backbone.config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=backbone.config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    if train_config.use_class_weights:
        crop_weights, disease_weights = _compute_class_weights(train_ds, head)
    else:
        crop_weights = torch.ones(len(head.crops))
        disease_weights = torch.ones(len(head.classes))

    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()),
        lr=backbone.config.learning_rate,
        weight_decay=backbone.config.weight_decay,
    )

    tracker = ExperimentTracker()
    run_name = run_name or f"{backbone_name}_seed{train_config.seed}"

    with tracker.start_run(run_name, configs={"train": train_config, "aug": aug_config}):
        for epoch in range(train_config.epochs):
            if epoch < backbone.config.freeze_backbone_epochs:
                backbone.freeze_backbone()
            else:
                backbone.unfreeze_backbone()

            train_metrics = _run_epoch(
                backbone, head, train_loader, train_config.device,
                crop_weights, disease_weights, train_config, optimizer,
            )  # fmt: skip
            val_metrics = _run_epoch(
                backbone, head, val_loader, train_config.device,
                crop_weights, disease_weights, train_config, optimizer=None,
            )  # fmt: skip

            tracker.log_metrics(
                {
                    "train_loss": train_metrics.loss,
                    "train_crop_acc": train_metrics.crop_accuracy,
                    "train_disease_acc": train_metrics.disease_accuracy,
                    "val_loss": val_metrics.loss,
                    "val_crop_acc": val_metrics.crop_accuracy,
                    "val_disease_acc": val_metrics.disease_accuracy,
                },
                step=epoch,
            )
            if on_epoch_end is not None:
                on_epoch_end(epoch, train_metrics, val_metrics)

        checkpoint_dir = train_config.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{run_name}.pt"
        torch.save(
            {
                "backbone_name": backbone_name,
                "backbone_state_dict": backbone.state_dict(),
                "head_state_dict": head.state_dict(),
            },
            checkpoint_path,
        )
        tracker.log_artifact(checkpoint_path)

    return backbone, head, checkpoint_path