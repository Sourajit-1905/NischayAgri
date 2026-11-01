import torch

from cropml.crop_rejection import CropRejectionConfig, CropRejectionGate
from cropml.data.label_space import LabelSpace
from cropml.models.hierarchical_head import HeadOutput, HierarchicalHead


def _label_space() -> LabelSpace:
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


def test_disease_prediction_always_matches_predicted_crop():
    label_space = _label_space()
    head = HierarchicalHead(embedding_dim=4, label_space=label_space)
    gate = CropRejectionGate(head, CropRejectionConfig(crop_confidence_threshold=0.0))

    # Deliberately make the RAW disease logits favor a tomato class, while
    # crop_logits strongly favor crop index 0 - proving masking overrides
    # an invalid raw disease argmax rather than trusting it directly.
    crop_logits = torch.tensor([[10.0, -10.0]])
    tomato_early_blight_idx = head.class_index("tomato__early_blight")
    disease_logits = torch.zeros(1, len(head.classes))
    disease_logits[0, tomato_early_blight_idx] = 10.0

    output = HeadOutput(
        crop_logits=crop_logits,
        disease_logits=disease_logits,
        crop_probs=torch.softmax(crop_logits, dim=-1),
        disease_probs=torch.softmax(disease_logits, dim=-1),
    )

    results = gate.apply(output)
    r = results[0]
    predicted_crop = head.crops[0]
    disease_crop = label_space.get(r.disease_class_id).crop
    assert disease_crop == predicted_crop


def test_low_confidence_crop_abstains():
    head = HierarchicalHead(embedding_dim=4, label_space=_label_space())
    gate = CropRejectionGate(head, CropRejectionConfig(crop_confidence_threshold=0.99))

    crop_logits = torch.tensor([[0.1, 0.0]])  # near-uniform -> low confidence
    disease_logits = torch.zeros(1, len(head.classes))
    output = HeadOutput(
        crop_logits=crop_logits,
        disease_logits=disease_logits,
        crop_probs=torch.softmax(crop_logits, dim=-1),
        disease_probs=torch.softmax(disease_logits, dim=-1),
    )

    results = gate.apply(output)
    assert results[0].crop_accepted is False
    assert results[0].disease_idx is None
    assert results[0].disease_class_id is None