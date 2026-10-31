"""
Unified label space shared by all datasets.

Tags are computed from dataset membership in label_space.yaml, never
hand-assigned, so they cannot drift out of sync with the mappings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).with_name("label_space.yaml")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# "<crop>__<condition>", exactly one double underscore.
_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*__[a-z0-9]+(?:_[a-z0-9]+)*$")


class LabelSpaceError(ValueError):
    """The label space definition is invalid."""


class UnmappedClassError(LabelSpaceError):
    """A dataset folder name has no entry in label_space.yaml."""


class ClassTag(StrEnum):
    IN_DOMAIN = "in_domain"
    SHARED = "shared"
    UNSEEN_IN_TRAINING = "unseen_in_training"
    AMBIGUOUS = "ambiguous"


class OodKind(StrEnum):
    UNSEEN_DISEASE_KNOWN_CROP = "unseen_disease_known_crop"
    UNSEEN_CROP = "unseen_crop"


def normalize_name(name: str) -> str:
    """Lowercase and collapse every non-alphanumeric run to one underscore."""
    return _NON_ALNUM.sub("_", name.lower()).strip("_")


@dataclass(frozen=True)
class LabelClass:
    id: str
    crop: str
    condition: str
    tag: ClassTag
    datasets: frozenset[str]
    ood_kind: OodKind | None

    @property
    def is_healthy(self) -> bool:
        return self.condition == "healthy"


class LabelSpace:
    def __init__(
        self,
        training_sources: tuple[str, ...],
        classes: tuple[LabelClass, ...],
        lookup: dict[str, dict[str, str]],
    ) -> None:
        self.training_sources = training_sources
        self.classes = classes
        self._by_id = {c.id: c for c in classes}
        self._lookup = lookup

    @classmethod
    def from_yaml(cls, path: Path | str = _DEFAULT_PATH) -> LabelSpace:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise LabelSpaceError(f"{path}: expected a mapping at the top level")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LabelSpace:
        missing = [k for k in ("training_sources", "classes", "datasets") if k not in raw]
        if missing:
            raise LabelSpaceError(f"missing required keys: {missing}")

        class_ids: list[str] = list(raw["classes"])
        bad_ids = [c for c in class_ids if not _ID_PATTERN.match(c)]
        if bad_ids:
            raise LabelSpaceError(f"class IDs must look like crop__condition: {bad_ids}")
        duplicates = sorted({c for c in class_ids if class_ids.count(c) > 1})
        if duplicates:
            raise LabelSpaceError(f"duplicate class IDs: {duplicates}")
        known = set(class_ids)

        ambiguous = set(raw.get("ambiguous") or [])
        if not ambiguous <= known:
            raise LabelSpaceError(f"ambiguous lists unknown classes: {sorted(ambiguous - known)}")

        datasets_raw: dict[str, dict[str, str]] = raw["datasets"]
        training_sources = tuple(raw["training_sources"])
        unknown_sources = [s for s in training_sources if s not in datasets_raw]
        if unknown_sources:
            raise LabelSpaceError(f"training_sources not in datasets: {unknown_sources}")

        lookup: dict[str, dict[str, str]] = {}
        membership: dict[str, set[str]] = {c: set() for c in class_ids}
        for ds_name, mapping in datasets_raw.items():
            table: dict[str, str] = {}
            for raw_name, class_id in mapping.items():
                if class_id not in known:
                    raise LabelSpaceError(
                        f"{ds_name}: '{raw_name}' maps to unknown class '{class_id}'"
                    )
                key = normalize_name(raw_name)
                if key in table and table[key] != class_id:
                    raise LabelSpaceError(
                        f"{ds_name}: '{raw_name}' collides with another name after "
                        f"normalization ('{key}')"
                    )
                table[key] = class_id
                membership[class_id].add(ds_name)
            lookup[ds_name] = table

        unreferenced = sorted(c for c, ds in membership.items() if not ds)
        if unreferenced:
            raise LabelSpaceError(f"classes not used by any dataset (typo?): {unreferenced}")

        training = set(training_sources)
        training_crops = {
            c.split("__")[0] for c in class_ids if c not in ambiguous and membership[c] & training
        }

        classes: list[LabelClass] = []
        for class_id in sorted(class_ids):
            ds = frozenset(membership[class_id])
            crop, condition = class_id.split("__")
            ood_kind: OodKind | None = None
            if class_id in ambiguous:
                tag = ClassTag.AMBIGUOUS
            elif ds & training:
                tag = ClassTag.SHARED if ds - training else ClassTag.IN_DOMAIN
            else:
                tag = ClassTag.UNSEEN_IN_TRAINING
                ood_kind = (
                    OodKind.UNSEEN_DISEASE_KNOWN_CROP
                    if crop in training_crops
                    else OodKind.UNSEEN_CROP
                )
            classes.append(LabelClass(class_id, crop, condition, tag, ds, ood_kind))

        return cls(training_sources, tuple(classes), lookup)

    def get(self, class_id: str) -> LabelClass:
        return self._by_id[class_id]

    def resolve(self, dataset: str, raw_name: str) -> LabelClass:
        """Map a dataset's folder name to its canonical class."""
        table = self._lookup.get(dataset)
        if table is None:
            raise LabelSpaceError(f"unknown dataset '{dataset}'")
        class_id = table.get(normalize_name(raw_name))
        if class_id is None:
            raise UnmappedClassError(f"{dataset}: no mapping for folder '{raw_name}'")
        return self._by_id[class_id]

    def find_unmapped(self, dataset: str, raw_names: list[str]) -> list[str]:
        """All folder names with no mapping, so they can be fixed in one pass."""
        table = self._lookup.get(dataset, {})
        return sorted(n for n in raw_names if normalize_name(n) not in table)

    def with_tag(self, tag: ClassTag) -> tuple[LabelClass, ...]:
        return tuple(c for c in self.classes if c.tag == tag)

    @property
    def training_classes(self) -> tuple[LabelClass, ...]:
        """The closed-set label space a model is trained on."""
        return tuple(c for c in self.classes if c.tag in (ClassTag.IN_DOMAIN, ClassTag.SHARED))

    @property
    def training_index(self) -> dict[str, int]:
        return {c.id: i for i, c in enumerate(self.training_classes)}

    def summary(self) -> dict[str, int]:
        return {tag.value: len(self.with_tag(tag)) for tag in ClassTag}


@lru_cache
def load_label_space() -> LabelSpace:
    return LabelSpace.from_yaml()


