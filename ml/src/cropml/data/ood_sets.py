"""
OOD evaluation set construction.

Pulls together everything label_space.py already knows about which
classes were never in the training label space, into ready-to-use
Sample subsets for OOD evaluation, plus an optional non-leaf negative set
sourced from outside the three plant datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cropml.data.label_space import ClassTag, OodKind
from cropml.data.loaders.base import Sample


class OodConfigError(ValueError):
    """
    The OOD set configuration is invalid.
    """


@dataclass(frozen=True)
class OodEvalSets:
    unseen_disease_known_crop: tuple[Sample, ...]
    unseen_crop: tuple[Sample, ...]
    non_leaf: tuple[Sample, ...]  # empty unless an external source is given

    def summary(self) -> dict[str, int]:
        return {
            "unseen_disease_known_crop": len(self.unseen_disease_known_crop),
            "unseen_crop": len(self.unseen_crop),
            "non_leaf": len(self.non_leaf),
        }


def build_ood_sets(
    all_samples: list[Sample],
    non_leaf_dir: Path | None = None,
    non_leaf_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> OodEvalSets:
    """
    Split out the unseen-in-training samples by OOD kind.

    all_samples should come from loading every configured dataset (not just
    the training source), so that unseen_in_training-tagged samples from
    PlantDoc/FieldPlant are actually present to select from.
    """
    from cropml.data.label_space import load_label_space

    label_space = load_label_space()

    unseen_disease: list[Sample] = []
    unseen_crop: list[Sample] = []
    for s in all_samples:
        label = label_space.get(s.class_id)
        if label.tag != ClassTag.UNSEEN_IN_TRAINING:
            continue
        if label.ood_kind == OodKind.UNSEEN_DISEASE_KNOWN_CROP:
            unseen_disease.append(s)
        elif label.ood_kind == OodKind.UNSEEN_CROP:
            unseen_crop.append(s)

    non_leaf: list[Sample] = []
    if non_leaf_dir is not None:
        if not non_leaf_dir.is_dir():
            raise OodConfigError(f"non_leaf_dir '{non_leaf_dir}' does not exist")
        for path in sorted(non_leaf_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in non_leaf_extensions:
                rel = path.relative_to(non_leaf_dir).as_posix()
                non_leaf.append(
                    Sample(
                        sample_id=f"non_leaf/{rel}",
                        path=path,
                        dataset="non_leaf",
                        domain="field",
                        class_id="__non_leaf__",
                        tag=ClassTag.UNSEEN_IN_TRAINING,
                        split=None,
                    )
                )

    return OodEvalSets(
        unseen_disease_known_crop=tuple(sorted(unseen_disease, key=lambda s: s.sample_id)),
        unseen_crop=tuple(sorted(unseen_crop, key=lambda s: s.sample_id)),
        non_leaf=tuple(non_leaf),
    )







