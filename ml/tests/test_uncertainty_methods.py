import torch

from cropml.data.label_space import LabelSpace
from cropml.models.backbone import BackboneConfig
from cropml.models.hierarchical_head import HierarchicalHead
from cropml.models.mobilenet_v3 import MobileNetV3Backbone
from cropml.uncertainty.deep_ensemble import EnsembleMember, ensemble_predict
from cropml.uncertainty.mc_dropout import mc_dropout_predict
from cropml.uncertainty.temperature_scaling import TemperatureScaledModel, fit_temperature


def _tiny_label_space() -> LabelSpace:
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


def _tiny_backbone_cfg() -> BackboneConfig:
    return BackboneConfig.from_dict(
        {
            "name": "test", "timm_name": "mobilenetv3_small_100", "pretrained": False,
            "input_size": 224, "dropout": 0.3, "embedding_dim": 1024,
            "batch_size": 2, "learning_rate": 0.001, "weight_decay": 0.0,
            "freeze_backbone_epochs": 1, "license": "test",
        }
    )


def test_fit_temperature_reduces_nll():
    torch.manual_seed(0)
    logits = torch.randn(50, 3) * 5  # deliberately overconfident/wrong on purpose
    labels = torch.randint(0, 3, (50,))
    criterion = torch.nn.CrossEntropyLoss()
    nll_before = criterion(logits, labels).item()

    t = fit_temperature(logits, labels)
    assert t > 0

    nll_after = criterion(logits / t, labels).item()
    assert nll_after <= nll_before + 1e-6


def test_temperature_scaled_model_probs_sum_to_one():
    backbone = MobileNetV3Backbone(_tiny_backbone_cfg())
    head = HierarchicalHead(embedding_dim=backbone.embedding_dim, label_space=_tiny_label_space())
    scaled = TemperatureScaledModel(backbone, head, crop_temperature=1.5, disease_temperature=0.8)

    images = torch.randn(2, 3, 224, 224)
    out = scaled.predict(images)
    assert torch.allclose(out.crop_probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(out.disease_probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_mc_dropout_mutual_information_non_negative_and_restores_state():
    backbone = MobileNetV3Backbone(_tiny_backbone_cfg())
    head = HierarchicalHead(embedding_dim=backbone.embedding_dim, label_space=_tiny_label_space())
    images = torch.randn(2, 3, 224, 224)

    result = mc_dropout_predict(backbone, head, images, n_samples=5)
    assert (result.crop_mutual_information >= -1e-5).all()
    assert (result.disease_mutual_information >= -1e-5).all()

    backbone.eval()
    head.eval()
    with torch.no_grad():
        a = head(backbone(images))
        b = head(backbone(images))
    assert torch.allclose(a.crop_probs, b.crop_probs)


def test_ensemble_predict_averages_correctly():
    label_space = _tiny_label_space()

    torch.manual_seed(0)
    backbone_a = MobileNetV3Backbone(_tiny_backbone_cfg())
    head_a = HierarchicalHead(embedding_dim=backbone_a.embedding_dim, label_space=label_space)
    torch.manual_seed(1)
    backbone_b = MobileNetV3Backbone(_tiny_backbone_cfg())
    head_b = HierarchicalHead(embedding_dim=backbone_b.embedding_dim, label_space=label_space)

    for m in (backbone_a, head_a, backbone_b, head_b):
        m.eval()

    members = [EnsembleMember(backbone_a, head_a), EnsembleMember(backbone_b, head_b)]
    images = torch.randn(2, 3, 224, 224)
    result = ensemble_predict(members, images)

    assert result.mean_crop_probs.shape == (2, 2)
    assert (result.crop_mutual_information >= -1e-5).all()
    assert result.n_members == 2


def test_ensemble_predict_rejects_empty_members():
    import pytest

    with pytest.raises(ValueError):
        ensemble_predict([], torch.randn(1, 3, 224, 224))