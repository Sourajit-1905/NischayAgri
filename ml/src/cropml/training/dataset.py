"""Bridges Phase 2.2's Sample/manifest pipeline into a torch Dataset.

Reads a split manifest (from manifest.py / build_splits.py's output) and
filters to one split only - training always uses exactly the leakage-free,
deduped split that was computed and persisted, never an ad-hoc re-split.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from cropml.data.label_space import load_label_space
from cropml.data.loaders import get_loader
from cropml.data.loaders.base import Sample
from cropml.data.manifest import load_manifest
from cropml.models.hierarchical_head import HierarchicalHead


class SplitDatasetError(ValueError):
    """The requested split/manifest combination is invalid."""


class CropDiseaseDataset(Dataset):
    """One dataset's samples, filtered to one split (train/val/test),
    with crop/disease label indices resolved via a HierarchicalHead's
    vocabulary (so the index space always matches the model being trained)."""

    def __init__(
        self,
        dataset_name: str,
        manifest_path: Path | str,
        split: str,
        head: HierarchicalHead,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.head = head
        self.transform = transform

        report = load_manifest(manifest_path)
        split_by_sample = report.as_dict_by_sample()

        all_samples: list[Sample] = get_loader(dataset_name).load()
        self.samples: list[Sample] = [
            s for s in all_samples if split_by_sample.get(s.sample_id) == split
        ]
        if not self.samples:
            raise SplitDatasetError(
                f"no samples found for dataset='{dataset_name}', split='{split}' - "
                f"check the manifest at '{manifest_path}' and that 'split' is one "
                f"of 'train', 'val', 'test'"
            )

        # Only samples whose class is actually in the head's training
        # vocabulary can be used for supervised training (e.g. a dataset
        # loaded here might include unseen_in_training/ambiguous classes
        # that aren't part of what the model is being trained to predict).
        trainable = [s for s in self.samples if s.class_id in head.class_to_idx]
        dropped = len(self.samples) - len(trainable)
        if dropped > 0:
            import warnings

            warnings.warn(
                f"{dataset_name}/{split}: dropped {dropped} sample(s) whose class "
                f"is not in the model's training vocabulary (e.g. unseen_in_training "
                f"or ambiguous classes)",
                UserWarning,
                stacklevel=2,
            )
        self.samples = trainable

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[object, int, int]:
        sample = self.samples[idx]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        label = self.head.label_space.get(sample.class_id)
        crop_idx = self.head.crop_index(label.crop)
        class_idx = self.head.class_index(sample.class_id)
        return image, crop_idx, class_idx

    def class_counts(self) -> dict[str, int]:
        """Per-class sample counts in THIS split - useful for computing
        class-weighted loss (flagged as needed, given PlantVillage's real
        class imbalance confirmed when the real data was loaded)."""
        counts: dict[str, int] = {}
        for s in self.samples:
            counts[s.class_id] = counts.get(s.class_id, 0) + 1
        return dict(sorted(counts.items()))