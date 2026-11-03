"""
Trains SeverityHead on cached pseudo-severity labels (step 3), using
embeddings from an already-trained Phase 2.3 backbone (frozen by default).

Training images are the normal 'color' variant - matching real inference
input - even though the pseudo-severity LABELS were derived from the
'segmented' variant (label-generation scaffolding only, never a model
input).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cropml.data.loaders import get_loader
from cropml.data.manifest import load_manifest
from cropml.models import get_backbone
from cropml.models.backbone import Backbone
from cropml.models.severity_head import SeverityHead, SeverityHeadConfig, coral_loss, decode_level
from cropml.training.augmentation import AugmentationConfig, build_eval_transform, build_train_transform

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "checkpoints"

LEVEL_TO_IDX: dict[str, int] = {"none": 0, "low": 1, "moderate": 2, "high": 3}
IDX_TO_LEVEL: dict[int, str] = {v: k for k, v in LEVEL_TO_IDX.items()}


class SeverityDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        manifest_path: Path | str,
        split: str,
        cache_path: Path | str,
        transform: Callable | None = None,
    ) -> None:
        self.transform = transform

        report = load_manifest(manifest_path)
        split_by_sample = report.as_dict_by_sample()
        all_samples = get_loader(dataset_name).load()

        with Path(cache_path).open("r", encoding="utf-8") as f:
            cache: dict[str, dict] = json.load(f)

        self.samples = []
        self.levels: list[int] = []
        skipped_no_cache = 0

        for sample in all_samples:
            if split_by_sample.get(sample.sample_id) != split:
                continue
            entry = cache.get(sample.sample_id)
            if entry is None:
                skipped_no_cache += 1
                continue
            self.samples.append(sample)
            self.levels.append(LEVEL_TO_IDX[entry["bucket"]])

        if skipped_no_cache > 0:
            warnings.warn(
                f"{dataset_name}/{split}: {skipped_no_cache} sample(s) missing from "
                f"severity cache (likely from the {skipped_no_cache} images with no "
                f"segmented counterpart, per compute_pseudo_severity.py), skipped",
                UserWarning,
                stacklevel=2,
            )
        if not self.samples:
            raise ValueError(f"no samples found for dataset='{dataset_name}', split='{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[object, int]:
        sample = self.samples[idx]
        with Image.open(sample.path) as img:  # the 'color' variant path - matches real inference input
            image = img.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, self.levels[idx]


@dataclass(frozen=True)
class SeverityTrainConfig:
    epochs: int = 10
    seed: int = 42
    device: str = "cpu"
    batch_size: int = 32
    learning_rate: float = 0.001
    num_workers: int = 0
    freeze_backbone: bool = True
    max_batches_per_epoch: int | None = None
    checkpoint_dir: Path = field(default_factory=lambda: _DEFAULT_CHECKPOINT_DIR)


@dataclass(frozen=True)
class SeverityEpochMetrics:
    loss: float
    accuracy: float
    mean_abs_error: float


def _run_severity_epoch(
    backbone: Backbone,
    severity_head: SeverityHead,
    loader: DataLoader,
    device: str,
    config: SeverityTrainConfig,
    optimizer: torch.optim.Optimizer | None,
) -> SeverityEpochMetrics:
    is_train = optimizer is not None
    severity_head.train(is_train)
    backbone.train(is_train and not config.freeze_backbone)

    total_loss = 0.0
    total_correct = 0
    total_abs_error = 0
    total_samples = 0

    backbone_context = torch.no_grad() if config.freeze_backbone else torch.enable_grad()
    outer_context = torch.enable_grad() if is_train else torch.no_grad()

    with outer_context:
        for batch_idx, (images, levels) in enumerate(loader):
            if config.max_batches_per_epoch is not None and batch_idx >= config.max_batches_per_epoch:
                break

            images = images.to(device)
            levels = levels.to(device)

            with backbone_context:
                embeddings = backbone(images)
            if config.freeze_backbone:
                embeddings = embeddings.detach()

            out = severity_head(embeddings)
            loss = coral_loss(out.cumulative_logits, levels, n_levels=severity_head.n_levels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            predicted = decode_level(out.cumulative_probs)
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (predicted == levels).sum().item()
            total_abs_error += (predicted - levels).abs().sum().item()
            total_samples += batch_size

    return SeverityEpochMetrics(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
        mean_abs_error=total_abs_error / max(total_samples, 1),
    )


def train_severity(
    backbone_name: str,
    dataset_name: str,
    manifest_path: Path | str,
    cache_path: Path | str,
    severity_head_config: SeverityHeadConfig,
    train_config: SeverityTrainConfig,
    aug_config: AugmentationConfig | None = None,
    backbone_checkpoint_path: Path | str | None = None,
    run_name: str | None = None,
    on_epoch_end: Callable[[int, SeverityEpochMetrics, SeverityEpochMetrics], None] | None = None,
) -> tuple[Backbone, SeverityHead, Path]:
    aug_config = aug_config or AugmentationConfig()
    torch.manual_seed(train_config.seed)

    backbone = get_backbone(backbone_name)
    if backbone_checkpoint_path is not None:
        checkpoint = torch.load(backbone_checkpoint_path, weights_only=True)
        backbone.load_state_dict(checkpoint["backbone_state_dict"])
    backbone.to(train_config.device)
    if train_config.freeze_backbone:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

    severity_head = SeverityHead(embedding_dim=backbone.embedding_dim, config=severity_head_config)
    severity_head.to(train_config.device)

    train_transform = build_train_transform(aug_config, backbone.config.input_size)
    eval_transform = build_eval_transform(backbone.config.input_size)
    train_ds = SeverityDataset(dataset_name, manifest_path, "train", cache_path, transform=train_transform)
    val_ds = SeverityDataset(dataset_name, manifest_path, "val", cache_path, transform=eval_transform)

    train_loader = DataLoader(
        train_ds, batch_size=train_config.batch_size, shuffle=True, num_workers=train_config.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_config.batch_size, shuffle=False, num_workers=train_config.num_workers
    )

    trainable_params = list(severity_head.parameters())
    if not train_config.freeze_backbone:
        trainable_params += list(backbone.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=train_config.learning_rate)

    for epoch in range(train_config.epochs):
        train_metrics = _run_severity_epoch(backbone, severity_head, train_loader, train_config.device, train_config, optimizer)
        val_metrics = _run_severity_epoch(backbone, severity_head, val_loader, train_config.device, train_config, optimizer=None)
        if on_epoch_end is not None:
            on_epoch_end(epoch, train_metrics, val_metrics)

    checkpoint_dir = train_config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name or f"{backbone_name}_severity_seed{train_config.seed}"
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    torch.save(
        {
            "backbone_name": backbone_name,
            "severity_head_config": severity_head_config.__dict__,
            "severity_head_state_dict": severity_head.state_dict(),
        },
        checkpoint_path,
    )

    return backbone, severity_head, checkpoint_path