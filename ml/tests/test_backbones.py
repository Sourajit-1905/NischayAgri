import pytest
import torch

from cropml.models.backbone import BackboneConfig
from cropml.models.efficientnet_b0 import EfficientNetB0Backbone
from cropml.models.mobilenet_v3 import MobileNetV3Backbone
from cropml.models.vit_small import ViTSmallBackbone


def _cfg(timm_name: str, embedding_dim: int, input_size: int = 224) -> BackboneConfig:
    return BackboneConfig.from_dict(
        {
            "name": "test", "timm_name": timm_name, "pretrained": False,
            "input_size": input_size, "dropout": 0.2, "embedding_dim": embedding_dim,
            "batch_size": 2, "learning_rate": 0.001, "weight_decay": 0.0,
            "freeze_backbone_epochs": 1, "license": "test",
        }
    )


@pytest.mark.parametrize(
    "backbone_cls,timm_name,embedding_dim",
    [
        (MobileNetV3Backbone, "mobilenetv3_small_100", 1024),
        (EfficientNetB0Backbone, "efficientnet_b0", 1280),
        (ViTSmallBackbone, "vit_small_patch16_224", 384),
    ],
)
def test_backbone_output_shape_matches_config(backbone_cls, timm_name, embedding_dim):
    cfg = _cfg(timm_name, embedding_dim)
    backbone = backbone_cls(cfg)
    backbone.eval()
    dummy = torch.randn(2, 3, cfg.input_size, cfg.input_size)
    with torch.no_grad():
        out = backbone(dummy)
    assert out.shape == (2, embedding_dim)


def test_wrong_embedding_dim_raises():
    # 999 does not match mobilenetv3_small_100's real forward-pass output -
    # regression test for the step 2/4 bug where a stale .num_features
    # value was trusted instead of a live forward pass.
    cfg = _cfg("mobilenetv3_small_100", 999)
    with pytest.raises(ValueError):
        MobileNetV3Backbone(cfg)


def test_mc_dropout_enable_disable_restores_state():
    cfg = _cfg("mobilenetv3_small_100", 1024)
    backbone = MobileNetV3Backbone(cfg)
    backbone.eval()
    dummy = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        out_a = backbone(dummy)
        out_b = backbone(dummy)
    assert torch.allclose(out_a, out_b)  # deterministic eval mode

    backbone.enable_mc_dropout()
    with torch.no_grad():
        mc_a = backbone(dummy)
        mc_b = backbone(dummy)
    backbone.disable_mc_dropout()
    assert not torch.allclose(mc_a, mc_b)  # genuinely stochastic during MC sampling

    with torch.no_grad():
        out_c = backbone(dummy)
        out_d = backbone(dummy)
    assert torch.allclose(out_c, out_d)  # restored to deterministic afterward