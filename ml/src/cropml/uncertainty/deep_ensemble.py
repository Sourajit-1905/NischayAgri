"""
Small deep ensemble: N independently-trained models, averaged at
inference. Same entropy/MI decomposition as mc_dropout.py, but
disagreement comes from genuinely different trained models rather than
within-model dropout stochasticity.
"""



from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from cropml.models import get_backbone
from cropml.models.backbone import Backbone
from cropml.models.hierarchical_head import HierarchicalHead
from cropml.training.augmentation import AugmentationConfig
from cropml.training.train_loop import TrainConfig, run_training


@dataclass(frozen=True)
class EnsembleMember:
    backbone: Backbone
    head: HierarchicalHead


@dataclass(frozen=True)
class EnsembleResult:
    mean_crop_probs: torch.Tensor
    mean_disease_probs: torch.Tensor
    crop_predictive_entropy: torch.Tensor
    crop_mutual_information: torch.Tensor
    disease_predictive_entropy: torch.Tensor
    disease_mutual_information: torch.Tensor
    n_members: int


def _entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(probs * (probs + eps).log()).sum(dim=-1)


def train_ensemble_members(
    backbone_name: str,
    dataset_name: str,
    manifest_path: Path | str,
    base_train_config: TrainConfig,
    n_members: int = 3,
    seeds: list[int] | None = None,
    aug_config: AugmentationConfig | None = None,
) -> list[Path]:
    """Trains n_members independent models (same architecture, different
    seeds), via run_training (step 10) unchanged. Returns their checkpoint
    paths. Each member is a FULL independent training run - expensive on
    CPU, budget accordingly."""
    seeds = seeds or list(range(n_members))
    if len(seeds) != n_members:
        raise ValueError(f"len(seeds)={len(seeds)} must equal n_members={n_members}")

    checkpoint_paths = []
    for seed in seeds:
        member_config = TrainConfig(
            epochs=base_train_config.epochs,
            seed=seed,
            device=base_train_config.device,
            num_workers=base_train_config.num_workers,
            use_class_weights=base_train_config.use_class_weights,
            lambda_crop=base_train_config.lambda_crop,
            lambda_disease=base_train_config.lambda_disease,
            max_batches_per_epoch=base_train_config.max_batches_per_epoch,
            log_every_n_steps=base_train_config.log_every_n_steps,
            checkpoint_dir=base_train_config.checkpoint_dir,
        )
        _, _, checkpoint_path = run_training(
            backbone_name=backbone_name,
            dataset_name=dataset_name,
            manifest_path=manifest_path,
            train_config=member_config,
            aug_config=aug_config or AugmentationConfig(),
            run_name=f"{backbone_name}_ensemble_seed{seed}",
        )
        checkpoint_paths.append(checkpoint_path)
    return checkpoint_paths


def load_ensemble_members(
    backbone_name: str, checkpoint_paths: list[Path]
) -> list[EnsembleMember]:
    """Loads N trained checkpoints (from train_ensemble_members, or any
    checkpoints saved in the format run_training/train_loop.py produces)
    into a list of EnsembleMember, ready for ensemble_predict."""
    members = []
    for path in checkpoint_paths:
        backbone = get_backbone(backbone_name)
        head = HierarchicalHead(embedding_dim=backbone.embedding_dim)
        checkpoint = torch.load(path, weights_only=True)
        backbone.load_state_dict(checkpoint["backbone_state_dict"])
        head.load_state_dict(checkpoint["head_state_dict"])
        backbone.eval()
        head.eval()
        members.append(EnsembleMember(backbone=backbone, head=head))
    return members


@torch.no_grad()
def ensemble_predict(members: list[EnsembleMember], images: torch.Tensor) -> EnsembleResult:
    if not members:
        raise ValueError("members list is empty - need at least one trained model")

    crop_probs_list = []
    disease_probs_list = []
    for member in members:
        embeddings = member.backbone(images)
        out = member.head(embeddings)
        crop_probs_list.append(out.crop_probs)
        disease_probs_list.append(out.disease_probs)

    crop_stack = torch.stack(crop_probs_list, dim=0)        # (n_members, batch, n_crops)
    disease_stack = torch.stack(disease_probs_list, dim=0)  # (n_members, batch, n_classes)

    mean_crop_probs = crop_stack.mean(dim=0)
    mean_disease_probs = disease_stack.mean(dim=0)

    crop_predictive_entropy = _entropy(mean_crop_probs)
    disease_predictive_entropy = _entropy(mean_disease_probs)

    crop_expected_entropy = _entropy(crop_stack).mean(dim=0)
    disease_expected_entropy = _entropy(disease_stack).mean(dim=0)

    crop_mi = crop_predictive_entropy - crop_expected_entropy
    disease_mi = disease_predictive_entropy - disease_expected_entropy

    return EnsembleResult(
        mean_crop_probs=mean_crop_probs,
        mean_disease_probs=mean_disease_probs,
        crop_predictive_entropy=crop_predictive_entropy,
        crop_mutual_information=crop_mi,
        disease_predictive_entropy=disease_predictive_entropy,
        disease_mutual_information=disease_mi,
        n_members=len(members),
    )




















