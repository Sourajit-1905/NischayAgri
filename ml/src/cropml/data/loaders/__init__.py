from __future__ import annotations

from pathlib import Path

from cropml.data.label_space import LabelSpace
from cropml.data.loaders.base import (
    BaseLoader,
    DatasetConfig,
    DatasetLayoutError,
    DatasetSanityWarning,
    LoaderConfigError,
    Sample,
    count_by_class,
)
from cropml.data.loaders.fieldplant import FieldPlantLoader
from cropml.data.loaders.plantdoc import PlantDocLoader
from cropml.data.loaders.plantvillage import PlantVillageLoader

LOADERS: dict[str, type[BaseLoader]] = {
    "plantvillage": PlantVillageLoader,
    "plantdoc": PlantDocLoader,
    "fieldplant": FieldPlantLoader,
}


def default_config_dir() -> Path:
    # ml/src/cropml/data/loaders/__init__.py -> ml/configs/data
    # Works with the editable install used in this project.
    return Path(__file__).resolve().parents[4] / "configs" / "data"


def get_loader(
    name: str,
    config_dir: Path | None = None,
    label_space: LabelSpace | None = None,
    strict: bool = True,
) -> BaseLoader:
    if name not in LOADERS:
        raise LoaderConfigError(f"unknown dataset '{name}'. Available: {sorted(LOADERS)}")
    path = (config_dir or default_config_dir()) / f"{name}.yaml"
    return LOADERS[name].from_yaml(path, label_space=label_space, strict=strict)


__all__ = [
    "LOADERS",
    "BaseLoader",
    "DatasetConfig",
    "DatasetLayoutError",
    "DatasetSanityWarning",
    "FieldPlantLoader",
    "LoaderConfigError",
    "PlantDocLoader",
    "PlantVillageLoader",
    "Sample",
    "count_by_class",
    "default_config_dir",
    "get_loader",
]

