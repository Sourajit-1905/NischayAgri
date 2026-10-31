"""
Shared dataset-loader machinery.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import OmegaConf

from cropml.data.label_space import ClassTag, LabelSpace, UnmappedClassError, load_label_space

Domain = Literal["lab", "field"]


class LoaderConfigError(ValueError):
    """
    The dataset config is invalid or asks for something unsupported.
    """


class DatasetLayoutError(FileNotFoundError):
    """
    The files on disk do not match the configured layout.
    """


class DatasetSanityWarning(UserWarning):
    """
    Loaded counts differ from what the config expects.
    """


@dataclass(frozen=True)
class ExpectedCounts:
    n_classes: int
    n_images_approx: int
    tolerance: float


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    domain: Domain
    root_dir: Path
    layout: str
    image_extensions: tuple[str, ...]
    group_strategy: str
    expected: ExpectedCounts
    license: str
    citation: str
    variant_subdir: str | None = None
    split_dirs: dict[str, str] | None = None
    group_metadata_key: str | None = None
    # Dataset-specific keys (e.g. allow_offline_augmented, annotation_format).
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> DatasetConfig:
        container = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(container, dict):
            raise LoaderConfigError(f"{path}: expected a mapping at the top level")
        return cls.from_dict({str(k): v for k, v in container.items()})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetConfig:
        required = (
            "name", "domain", "root_dir", "layout", "image_extensions",
            "group_strategy", "expected", "license", "citation",
        )  # fmt: skip
        missing = [k for k in required if k not in raw]
        if missing:
            raise LoaderConfigError(f"missing required config keys: {missing}")
        if raw["domain"] not in ("lab", "field"):
            raise LoaderConfigError(f"domain must be 'lab' or 'field', got {raw['domain']!r}")

        exts = tuple(
            (e if e.startswith(".") else f".{e}").lower() for e in raw["image_extensions"]
        )
        expected = ExpectedCounts(
            n_classes=int(raw["expected"]["n_classes"]),
            n_images_approx=int(raw["expected"]["n_images_approx"]),
            tolerance=float(raw["expected"]["tolerance"]),
        )
        known = set(required) | {"variant_subdir", "split_dirs", "group_metadata_key"}
        return cls(
            name=str(raw["name"]),
            domain=cast(Domain, raw["domain"]),
            root_dir=Path(str(raw["root_dir"])),
            layout=str(raw["layout"]),
            image_extensions=exts,
            group_strategy=str(raw["group_strategy"]),
            expected=expected,
            license=str(raw["license"]),
            citation=str(raw["citation"]),
            variant_subdir=raw.get("variant_subdir"),
            split_dirs=raw.get("split_dirs"),
            group_metadata_key=raw.get("group_metadata_key"),
            options={k: v for k, v in raw.items() if k not in known},
        )


@dataclass(frozen=True)
class Sample:
    sample_id: str  # "<dataset>/<posix path relative to root_dir>"
    path: Path
    dataset: str
    domain: Domain
    class_id: str
    tag: ClassTag
    split: str | None  # the dataset's own official split, if it has one


@dataclass(frozen=True)
class RawItem:
    path: Path
    raw_class: str
    split: str | None


def count_by_class(samples: list[Sample]) -> dict[str, int]:
    return dict(sorted(Counter(s.class_id for s in samples).items()))


class BaseLoader(ABC):
    def __init__(
        self,
        config: DatasetConfig,
        label_space: LabelSpace | None = None,
        strict: bool = True,
    ) -> None:
        self.config = config
        self.label_space = label_space or load_label_space()
        self.strict = strict

    @classmethod
    def from_yaml(
        cls,
        path: Path | str,
        label_space: LabelSpace | None = None,
        strict: bool = True,
    ) -> BaseLoader:
        return cls(DatasetConfig.from_yaml(path), label_space, strict)

    @abstractmethod
    def _discover(self) -> Iterator[RawItem]:
        """Yield every image with its raw class folder name and official split."""

    def load(self) -> list[Sample]:
        cfg = self.config
        if not cfg.root_dir.is_dir():
            raise DatasetLayoutError(
                f"{cfg.name}: root_dir '{cfg.root_dir}' does not exist. Set the "
                "CROPML_DATA_ROOT environment variable or edit the config."
            )

        items = sorted(self._discover(), key=lambda i: (i.split or "", i.path.as_posix()))
        if not items:
            raise DatasetLayoutError(
                f"{cfg.name}: no images found under '{cfg.root_dir}'. "
                f"Check root_dir and layout ('{cfg.layout}')."
            )

        raw_names = sorted({i.raw_class for i in items})
        unmapped = self.label_space.find_unmapped(cfg.name, raw_names)
        if unmapped:
            msg = f"{cfg.name}: {len(unmapped)} folder name(s) have no label mapping: {unmapped}"
            if self.strict:
                raise UnmappedClassError(msg)
            warnings.warn(msg + " (skipped)", DatasetSanityWarning, stacklevel=2)
            skip = set(unmapped)
            items = [i for i in items if i.raw_class not in skip]

        samples: list[Sample] = []
        for item in items:
            label = self.label_space.resolve(cfg.name, item.raw_class)
            rel = item.path.relative_to(cfg.root_dir).as_posix()
            samples.append(
                Sample(
                    sample_id=f"{cfg.name}/{rel}",
                    path=item.path,
                    dataset=cfg.name,
                    domain=cfg.domain,
                    class_id=label.id,
                    tag=label.tag,
                    split=item.split,
                )
            )
        self._sanity_check(samples)
        return samples

    #   helpers for subclasses  

    def _find_subdir_ci(self, parent: Path, name: str) -> Path:
        """Find a subdirectory by name, ignoring case (e.g. 'train' vs 'TRAIN')."""
        for child in sorted(parent.iterdir()):
            if child.is_dir() and child.name.lower() == name.lower():
                return child
        available = sorted(p.name for p in parent.iterdir() if p.is_dir())
        raise DatasetLayoutError(
            f"{self.config.name}: no '{name}' folder in '{parent}'. Found: {available}"
        )

    def _scan_class_folders(self, base_dir: Path, split: str | None) -> Iterator[RawItem]:
        """Treat each immediate subfolder of base_dir as one class."""
        exts = self.config.image_extensions
        for class_dir in sorted(base_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith((".", "__")):
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in exts:
                    yield RawItem(path=path, raw_class=class_dir.name, split=split)

    def _sanity_check(self, samples: list[Sample]) -> None:
        exp = self.config.expected
        n_classes = len({s.class_id for s in samples})
        if n_classes != exp.n_classes:
            warnings.warn(
                f"{self.config.name}: found {n_classes} classes, expected {exp.n_classes}",
                DatasetSanityWarning,
                stacklevel=3,
            )
        drift = abs(len(samples) - exp.n_images_approx) / exp.n_images_approx
        if drift > exp.tolerance:
            warnings.warn(
                f"{self.config.name}: found {len(samples)} images, expected about "
                f"{exp.n_images_approx} (tolerance {exp.tolerance:.0%})",
                DatasetSanityWarning,
                stacklevel=3,
            )