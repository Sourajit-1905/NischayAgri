import pytest
import torch

from cropml.data.label_space import LabelSpace
from cropml.models.hierarchical_head import HierarchicalHead


def _small_label_space() -> LabelSpace:
    return LabelSpace.from_dict(
        {
            "training_sources": ["lab"],
            "classes": ["tomato__healthy", "tomato__early_blight", "potato__healthy"],
            "ambiguous": [],
            "datasets": {
                "lab": {
                    "Tomato___healthy": "tomato__healthy",
                    "Tomato___Early_blight": "tomato__early_blight",
                    "Potato___healthy": "potato__healthy",
                }
            },
        }
    )


def test_head_output_shapes_and_softmax():
    head = HierarchicalHead(embedding_dim=16, label_space=_small_label_space())
    assert len(head.crops) == 2
    assert len(head.classes) == 3

    embeddings = torch.randn(4, 16)
    out = head(embeddings)
    assert out.crop_logits.shape == (4, 2)
    assert out.disease_logits.shape == (4, 3)
    assert torch.allclose(out.crop_probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(out.disease_probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_decode_round_trip():
    head = HierarchicalHead(embedding_dim=8, label_space=_small_label_space())
    crop_idx = head.crop_index("tomato")
    class_idx = head.class_index("tomato__early_blight")
    crop_name, class_id = head.decode(crop_idx, class_idx)
    assert crop_name == "tomato"
    assert class_id == "tomato__early_blight"


def test_unknown_crop_raises():
    head = HierarchicalHead(embedding_dim=8, label_space=_small_label_space())
    with pytest.raises(KeyError):
        head.crop_index("unknown_crop")


def test_no_training_classes_raises():
    # tomato__healthy is only mapped under "other", not under the declared
    # training_sources=["train_src"] - so it ends up unseen_in_training,
    # leaving training_classes empty.
    raw = {
        "training_sources": ["train_src"],
        "classes": ["tomato__healthy"],
        "ambiguous": [],
        "datasets": {
            "train_src": {},
            "other": {"Tomato___healthy": "tomato__healthy"},
        },
    }
    label_space = LabelSpace.from_dict(raw)
    with pytest.raises(ValueError):
        HierarchicalHead(embedding_dim=8, label_space=label_space)