import pytest
import torch

from cropml.data.label_space import LabelSpace
from cropml.models.backbone import BackboneConfig
from cropml.models.hierarchical_head import HierarchicalHead
from cropml.models.mobilenet_v3 import MobileNetV3Backbone
from cropml.ood.energy import energy_score
from cropml.ood.mahalanobis import MahalanobisFitError, fit_mahalanobis, mahalanobis_score
from cropml.ood.msp import msp_score
from cropml.ood.odin import odin_score


def test_msp_score_direction():
    confident = torch.tensor([[0.95, 0.03, 0.02]])
    uncertain = torch.tensor([[0.34, 0.33, 0.33]])
    assert msp_score(uncertain).item() > msp_score(confident).item()


def test_energy_score_direction():
    confident_logits = torch.tensor([[10.0, 0.0, 0.0]])
    uncertain_logits = torch.tensor([[1.0, 1.0, 1.0]])
    assert energy_score(uncertain_logits).item() > energy_score(confident_logits).item()


def test_mahalanobis_distance_ordering():
    torch.manual_seed(0)
    class_a = torch.randn(30, 2) * 0.3
    class_b = torch.randn(30, 2) * 0.3 + torch.tensor([5.0, 5.0])
    embeddings = torch.cat([class_a, class_b])
    labels = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)])

    params = fit_mahalanobis(embeddings, labels)
    near = mahalanobis_score(torch.tensor([[0.0, 0.0]]), params)
    far = mahalanobis_score(torch.tensor([[50.0, -50.0]]), params)
    assert far.item() > near.item()


def test_mahalanobis_insufficient_samples_raises():
    embeddings = torch.randn(1, 2)
    labels = torch.zeros(1, dtype=torch.long)
    with pytest.raises(MahalanobisFitError):
        fit_mahalanobis(embeddings, labels)


def _tiny_odin_setup():
    label_space = LabelSpace.from_dict(
        {
            "training_sources": ["lab"],
            "classes": ["tomato__healthy", "tomato__early_blight"],
            "ambiguous": [],
            "datasets": {
                "lab": {
                    "Tomato___healthy": "tomato__healthy",
                    "Tomato___Early_blight": "tomato__early_blight",
                }
            },
        }
    )
    cfg = BackboneConfig.from_dict(
        {
            "name": "test", "timm_name": "mobilenetv3_small_100", "pretrained": False,
            "input_size": 224, "dropout": 0.2, "embedding_dim": 1024,
            "batch_size": 2, "learning_rate": 0.001, "weight_decay": 0.0,
            "freeze_backbone_epochs": 1, "license": "test",
        }
    )
    backbone = MobileNetV3Backbone(cfg)
    head = HierarchicalHead(embedding_dim=backbone.embedding_dim, label_space=label_space)
    backbone.eval()
    head.eval()
    return backbone, head


def test_odin_score_in_valid_range_and_does_not_mutate_input():
    backbone, head = _tiny_odin_setup()

    def forward_fn(images: torch.Tensor) -> torch.Tensor:
        return head(backbone(images)).crop_logits

    images = torch.randn(2, 3, 224, 224)
    original = images.clone()

    scores = odin_score(forward_fn, images, temperature=1000.0, epsilon=0.0014)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert torch.equal(images, original)





    