import pytest

from cropml.data.label_space import (
    ClassTag,
    LabelSpace,
    LabelSpaceError,
    OodKind,
    UnmappedClassError,
    load_label_space,
    normalize_name,
)


def _minimal_raw() -> dict:
    return {
        "training_sources": ["lab"],
        "classes": ["tomato__healthy", "tomato__early_blight", "corn__smut"],
        "ambiguous": [],
        "datasets": {
            "lab": {
                "Tomato___healthy": "tomato__healthy",
                "Tomato___Early_blight": "tomato__early_blight",
            },
            "field": {
                "Tomato healthy leaf": "tomato__healthy",
                "Corn smut": "corn__smut",
            },
        },
    }


def test_normalize_name_collapses_punctuation_and_case():
    assert normalize_name("Tomato___Early_blight") == normalize_name("tomato early blight")


def test_tags_computed_correctly():
    ls = LabelSpace.from_dict(_minimal_raw())
    assert ls.get("tomato__healthy").tag == ClassTag.SHARED  # in lab (training) + field
    assert ls.get("tomato__early_blight").tag == ClassTag.IN_DOMAIN  # lab only
    assert ls.get("corn__smut").tag == ClassTag.UNSEEN_IN_TRAINING  # field only, unseen crop
    assert ls.get("corn__smut").ood_kind == OodKind.UNSEEN_CROP


def test_unseen_disease_known_crop_detected():
    raw = _minimal_raw()
    raw["classes"].append("tomato__mosaic_virus")
    raw["datasets"]["field"]["Tomato mosaic virus leaf"] = "tomato__mosaic_virus"
    ls = LabelSpace.from_dict(raw)
    label = ls.get("tomato__mosaic_virus")
    assert label.tag == ClassTag.UNSEEN_IN_TRAINING
    assert label.ood_kind == OodKind.UNSEEN_DISEASE_KNOWN_CROP  # tomato IS a training crop


def test_ambiguous_classes_excluded_from_shared_and_ood():
    raw = _minimal_raw()
    raw["classes"].append("pepper__leaf_spot")
    raw["ambiguous"] = ["pepper__leaf_spot"]
    raw["datasets"]["field"]["Pepper leaf spot"] = "pepper__leaf_spot"
    ls = LabelSpace.from_dict(raw)
    label = ls.get("pepper__leaf_spot")
    assert label.tag == ClassTag.AMBIGUOUS
    assert label.ood_kind is None


def test_resolve_is_case_and_punctuation_insensitive():
    ls = LabelSpace.from_dict(_minimal_raw())
    assert ls.resolve("field", "tomato healthy leaf").id == "tomato__healthy"
    assert ls.resolve("field", "TOMATO___HEALTHY___LEAF").id == "tomato__healthy"


def test_resolve_unmapped_raises():
    ls = LabelSpace.from_dict(_minimal_raw())
    with pytest.raises(UnmappedClassError):
        ls.resolve("field", "Some Unknown Class")


def test_find_unmapped_lists_all_misses_at_once():
    ls = LabelSpace.from_dict(_minimal_raw())
    names = ["Tomato healthy leaf", "Totally Unknown A", "Totally Unknown B"]
    assert ls.find_unmapped("field", names) == ["Totally Unknown A", "Totally Unknown B"]


def test_class_id_must_match_crop_condition_pattern():
    raw = _minimal_raw()
    raw["classes"][0] = "not-a-valid-id"
    with pytest.raises(LabelSpaceError):
        LabelSpace.from_dict(raw)


def test_duplicate_class_ids_rejected():
    raw = _minimal_raw()
    raw["classes"].append(raw["classes"][0])
    with pytest.raises(LabelSpaceError):
        LabelSpace.from_dict(raw)


def test_mapping_to_unknown_class_rejected():
    raw = _minimal_raw()
    raw["datasets"]["field"]["Something odd"] = "nonexistent__class"
    with pytest.raises(LabelSpaceError):
        LabelSpace.from_dict(raw)


def test_class_with_no_dataset_membership_rejected():
    raw = _minimal_raw()
    raw["classes"].append("orphan__class")  # never referenced by any dataset mapping
    with pytest.raises(LabelSpaceError):
        LabelSpace.from_dict(raw)


def test_training_classes_excludes_unseen_and_ambiguous():
    raw = _minimal_raw()
    raw["classes"].append("pepper__leaf_spot")
    raw["ambiguous"] = ["pepper__leaf_spot"]
    raw["datasets"]["field"]["Pepper leaf spot"] = "pepper__leaf_spot"
    ls = LabelSpace.from_dict(raw)
    training_ids = {c.id for c in ls.training_classes}
    assert training_ids == {"tomato__healthy", "tomato__early_blight"}
    assert "corn__smut" not in training_ids  # unseen_in_training
    assert "pepper__leaf_spot" not in training_ids  # ambiguous


def test_real_label_space_yaml_loads_without_error():
    """
    Smoke test against the actual project label_space.yaml (step 3).
    """
    ls = load_label_space()
    assert len(ls.classes) > 0
    assert len(ls.training_classes) > 0