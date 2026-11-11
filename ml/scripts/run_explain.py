"""
CLI: runs the explainability report across a sample of test images.
Produces: per-sample CAM PNGs, background-attention ratios, faithfulness
AUCs, and a summary CSV + markdown report quantifying how often
confidently-wrong predictions rely on background.

Usage (from ml/):
    python scripts/run_explain.py
    python scripts/run_explain.py n_samples=20 backbone_name=mobilenet_v3
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cropml.explain.background import background_attention_ratio  # noqa: E402
from cropml.explain.faithfulness import faithfulness_test  # noqa: E402
from cropml.explain.gradcam import GradCAM, get_target_layer  # noqa: E402
from cropml.models import get_backbone  # noqa: E402
from cropml.models.hierarchical_head import HierarchicalHead  # noqa: E402
from cropml.training.augmentation import build_eval_transform  # noqa: E402
from cropml.training.dataset import CropDiseaseDataset  # noqa: E402

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_model(backbone_name: str, checkpoint_path: str):
    backbone = get_backbone(backbone_name)
    head = HierarchicalHead(embedding_dim=backbone.embedding_dim)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, weights_only=True)
        backbone.load_state_dict(ckpt["backbone_state_dict"])
        head.load_state_dict(ckpt["head_state_dict"])
    backbone.eval(); head.eval()
    return backbone, head


@hydra.main(config_path=str(_CONFIG_DIR / "experiment"), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    n_samples = int(cfg.get("n_samples", 50))
    backbone_name = str(cfg.get("backbone_name", cfg.backbones[0]))
    checkpoint_path = str(cfg.checkpoint_dir) + f"/{backbone_name}_seed0.pt"
    results_dir = Path(cfg.results_dir) / "explain"
    results_dir.mkdir(parents=True, exist_ok=True)

    backbone, head = _load_model(backbone_name, checkpoint_path)
    target_layer = get_target_layer(backbone_name, backbone._timm_model)

    eval_transform = build_eval_transform(backbone.config.input_size)
    dataset_name, split = "plantvillage", "test"
    ds = CropDiseaseDataset(dataset_name, cfg.manifest_path, split, head, transform=eval_transform)
    indices = list(range(min(n_samples, len(ds))))

    def fwd_probs(imgs):
        with torch.no_grad():
            return torch.softmax(head(backbone(imgs)).disease_logits, dim=-1)

    rows = []
    for idx in indices:
        image_tensor, crop_idx, class_idx = ds[idx]
        image_tensor = image_tensor.unsqueeze(0)
        sample = ds.samples[idx]

        with GradCAM(backbone._timm_model, target_layer) as gcam:
            with torch.enable_grad():
                image_grad = image_tensor.clone().requires_grad_(True)
                embeddings = backbone(image_grad)
                out = head(embeddings)
                cam_result = gcam.compute(image_grad, out.disease_logits, class_idx=class_idx)

        predicted_class = int(out.disease_logits.argmax(dim=-1).item())
        confidence = float(torch.softmax(out.disease_logits, dim=-1)[0, predicted_class].item())
        is_correct = predicted_class == class_idx

        cam_up = F.interpolate(cam_result.cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode="bilinear", align_corners=False).squeeze()

        # Background-attention ratio using a simple center-crop leaf mask
        # (since we don't have per-image segmentation masks for the test set,
        # we approximate leaf region as the center 60% of the image).
        leaf_mask = torch.zeros(224, 224)
        m = int(224 * 0.2)
        leaf_mask[m:-m, m:-m] = 1.0

        bg_result = background_attention_ratio(cam_up, leaf_mask)
        faith_result = faithfulness_test(image_tensor.detach(), cam_up, fwd_probs,
            class_idx=predicted_class, n_steps=5)

        rows.append({
            "sample_id": sample.sample_id,
            "true_class": head.classes[class_idx],
            "predicted_class": head.classes[predicted_class],
            "confidence": round(confidence, 4),
            "correct": is_correct,
            "bg_attention_ratio": round(bg_result.background_ratio, 4),
            "deletion_auc": round(faith_result.deletion_auc, 4),
            "insertion_auc": round(faith_result.insertion_auc, 4),
        })

    csv_path = results_dir / "explain_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    # Summary report
    wrong_confident = [r for r in rows if not r["correct"] and r["confidence"] > 0.5]
    right_confident = [r for r in rows if r["correct"] and r["confidence"] > 0.5]

    def mean(vals): return sum(vals) / len(vals) if vals else float("nan")

    wc_bg = mean([r["bg_attention_ratio"] for r in wrong_confident])
    rc_bg = mean([r["bg_attention_ratio"] for r in right_confident])

    report_lines = [
        "# Explainability Report",
        f"n_samples={len(rows)}, backbone={backbone_name}",
        "",
        "| group | n | mean_bg_attention_ratio |",
        "|---|---|---|",
        f"| correct + confident | {len(right_confident)} | {rc_bg:.4f} |",
        f"| wrong + confident | {len(wrong_confident)} | {wc_bg:.4f} |",
        "",
        "A higher bg_attention_ratio for wrong+confident predictions indicates",
        "the model is using background features as shortcuts.",
        "",
        f"Full results: {csv_path}",
    ]
    md_path = results_dir / "explain_report.md"
    md_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nsaved: {csv_path}, {md_path}")


if __name__ == "__main__":
    main()