import pytest

from cropml.data.label_space import LabelSpace, UnmappedClassError
from cropml.data.loaders.base import (
    DatasetConfig,
    DatasetLayoutError,
    DatasetSanityWarning,
    LoaderConfigError,
)
from cropml.data.loaders.plantvillage import PlantVillageLoader


def _label_space_for_two_classes() -> LabelSpace:
    return LabelSpace.from_dict(
        {
            "training_sources": ["plantvillage"],
            "classes": ["tomato__healthy", "tomato__early_blight"],
            "ambiguous": [],
            "datasets": {
                "plantvillage": {
                    "Tomato___healthy": "tomato__healthy",
                    "Tomato___Early_blight": "tomato__early_blight",
                }
            },
        }
    )


def _config(root_dir, **overrides) -> DatasetConfig:
    base = {
        "name": "plantvillage",
        "domain": "lab",
        "root_dir": str(root_dir),
        "layout": "class_folders",
        "image_extensions": [".jpg"],
        "group_strategy": "phash_cluster",
        "expected": {"n_classes": 2, "n_images_approx": 4, "tolerance": 0.5},
        "license": "test",
        "citation": "test",
        "variant_subdir": "color",
    }
    base.update(overrides)
    return DatasetConfig.from_dict(base)


def test_loads_and_maps_known_classes(tmp_data_root, make_image_file):
    root = tmp_data_root / "plantvillage"
    make_image_file(root / "color" / "Tomato___healthy" / "a.jpg", seed=1)
    make_image_file(root / "color" / "Tomato___Early_blight" / "b.jpg", seed=2)

    loader = PlantVillageLoader(_config(root), label_space=_label_space_for_two_classes())
    samples = loader.load()

    assert {s.class_id for s in samples} == {"tomato__healthy", "tomato__early_blight"}


def test_missing_root_dir_raises_layout_error(tmp_data_root):
    loader = PlantVillageLoader(
        _config(tmp_data_root / "does_not_exist"), label_space=_label_space_for_two_classes()
    )
    with pytest.raises(DatasetLayoutError):
        loader.load()


def test_unmapped_class_raises_in_strict_mode(tmp_data_root, make_image_file):
    root = tmp_data_root / "plantvillage"
    make_image_file(root / "color" / "Tomato___healthy" / "a.jpg", seed=1)
    make_image_file(root / "color" / "Unknown___Disease" / "b.jpg", seed=2)

    loader = PlantVillageLoader(_config(root), label_space=_label_space_for_two_classes(), strict=True)
    with pytest.raises(UnmappedClassError):
        loader.load()


def test_unmapped_class_skipped_in_non_strict_mode(tmp_data_root, make_image_file):
    root = tmp_data_root / "plantvillage"
    make_image_file(root / "color" / "Tomato___healthy" / "a.jpg", seed=1)
    make_image_file(root / "color" / "Unknown___Disease" / "b.jpg", seed=2)

    loader = PlantVillageLoader(_config(root), label_space=_label_space_for_two_classes(), strict=False)
    with pytest.warns(DatasetSanityWarning):
        samples = loader.load()
    assert {s.class_id for s in samples} == {"tomato__healthy"}


def test_offline_augmented_folder_name_is_rejected(tmp_data_root, make_image_file):
    root = tmp_data_root / "plantvillage_augmented"
    make_image_file(root / "color" / "Tomato___healthy" / "a.jpg", seed=1)

    loader = PlantVillageLoader(_config(root), label_space=_label_space_for_two_classes())
    with pytest.raises(LoaderConfigError):
        loader.load()


def test_load_is_deterministically_ordered(tmp_data_root, make_image_file):
    root = tmp_data_root / "plantvillage"
    for i in range(5):
        make_image_file(root / "color" / "Tomato___healthy" / f"img{i}.jpg", seed=i)

    cfg = _config(root, expected={"n_classes": 1, "n_images_approx": 5, "tolerance": 0.5})
    ls = _label_space_for_two_classes()

    ids_a = [s.sample_id for s in PlantVillageLoader(cfg, label_space=ls).load()]
    ids_b = [s.sample_id for s in PlantVillageLoader(cfg, label_space=ls).load()]
    assert ids_a == ids_b == sorted(ids_a)