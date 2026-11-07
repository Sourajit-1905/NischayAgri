import base64
import io

import pytest
import torch
import torch.nn.functional as F

from cropml.explain.background import background_attention_ratio, masking_experiment
from cropml.explain.faithfulness import faithfulness_test
from cropml.explain.gradcam import GradCAM, GradCAMError, get_target_layer
from cropml.explain.heatmap import cam_to_base64_png, cam_to_heatmap_pil
from cropml.explain.scorecam import score_cam
from cropml.models.backbone import BackboneConfig
from cropml.models.mobilenet_v3 import MobileNetV3Backbone
from PIL import Image


def _backbone():
    cfg = BackboneConfig.from_dict({
        "name": "test", "timm_name": "mobilenetv3_small_100", "pretrained": False,
        "input_size": 224, "dropout": 0.0, "embedding_dim": 1024,
        "batch_size": 2, "learning_rate": 0.001, "weight_decay": 0.0,
        "freeze_backbone_epochs": 1, "license": "test",
    })
    m = MobileNetV3Backbone(cfg)
    m.eval()
    return m


#   Grad-CAM  

def test_gradcam_output_shape_and_range():
    backbone = _backbone()
    target_layer = get_target_layer("mobilenet_v3", backbone._timm_model)
    image = torch.randn(1, 3, 224, 224).requires_grad_(True)
    with GradCAM(backbone._timm_model, target_layer) as gcam:
        with torch.enable_grad():
            # logits must come from a real forward pass through the backbone
            # so gradients can flow back through target_layer on backward.
            embeddings = backbone(image)
            logits = torch.nn.Linear(embeddings.shape[-1], 10)(embeddings)
            result = gcam.compute(image, logits, class_idx=0)
    assert result.cam.ndim == 2
    assert result.cam.min() >= 0 and result.cam.max() <= 1


def test_gradcam_vit_raises():
    with pytest.raises(GradCAMError):
        get_target_layer("vit_small", torch.nn.Linear(1, 1))


def test_gradcam_no_activation_raises():
    backbone = _backbone()
    target_layer = get_target_layer("mobilenet_v3", backbone._timm_model)
    gcam = GradCAM(backbone._timm_model, target_layer)
    with pytest.raises(GradCAMError):
        gcam.compute(torch.randn(1, 3, 224, 224), torch.randn(1, 10), class_idx=0)


#   Score-CAM  

def test_scorecam_output_shape_and_range():
    backbone = _backbone()
    target_layer = get_target_layer("mobilenet_v3", backbone._timm_model)
    image = torch.randn(1, 3, 224, 224)

    def fwd(img):
        with torch.no_grad():
            return backbone(img)

    result = score_cam(image, target_layer, fwd, max_channels=4)
    assert result.cam.shape == (224, 224)
    assert result.cam.min() >= 0 and result.cam.max() <= 1


#   Heatmap  

def test_cam_to_base64_png_valid():
    cam = torch.rand(7, 7)
    b64 = cam_to_base64_png(cam, target_size=(224, 224))
    decoded = base64.b64decode(b64)
    assert decoded[:8] == bytes([137, 80, 78, 71, 13, 10, 26, 10])
    img = Image.open(io.BytesIO(decoded))
    assert img.mode == "RGB"


def test_cam_to_heatmap_pil_with_overlay():
    cam = torch.rand(7, 7)
    overlay = Image.new("RGB", (224, 224), color=(100, 150, 100))
    result = cam_to_heatmap_pil(cam, (224, 224), overlay_image=overlay, alpha=0.5)
    assert result.size == (224, 224)


#  Faithfulness 

def test_faithfulness_curves_correct_length():
    cam = torch.rand(7, 7)
    image = torch.randn(1, 3, 7, 7)

    def fwd(img):
        return torch.softmax(torch.randn(img.shape[0], 5), dim=-1)

    result = faithfulness_test(image, cam, fwd, class_idx=0, n_steps=4)
    assert len(result.deletion_curve) == 4
    assert len(result.insertion_curve) == 4
    assert 0 <= result.deletion_auc
    assert 0 <= result.insertion_auc


#  Background 

def test_background_ratio_all_foreground():
    cam = torch.ones(7, 7)
    mask = torch.ones(7, 7)
    r = background_attention_ratio(cam, mask)
    assert abs(r.background_ratio) < 1e-5
    assert abs(r.foreground_ratio - 1.0) < 1e-5


def test_background_ratio_all_background():
    cam = torch.ones(7, 7)
    mask = torch.zeros(7, 7)
    r = background_attention_ratio(cam, mask)
    assert abs(r.background_ratio - 1.0) < 1e-5


def test_masking_experiment_shape():
    images = torch.randn(3, 3, 32, 32)
    masks = torch.ones(3, 1, 32, 32)

    def fwd(imgs):
        return torch.softmax(torch.randn(imgs.shape[0], 5), dim=-1)

    r = masking_experiment(images, masks, fwd, class_indices=[0, 1, 2])
    assert r.n_samples == 3
    assert isinstance(r.mean_confidence_drop, float)



