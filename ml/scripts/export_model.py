"""
CLI: export a trained model to ONNX, run parity test, write manifest.

Usage (from ml/):
    python scripts/export_model.py backbone_name=mobilenet_v3 seed=0
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cropml.export.onnx_export import ExportConfig, export_to_onnx  # noqa: E402
from cropml.export.parity_test import check_parity  # noqa: E402
from cropml.models import get_backbone  # noqa: E402
from cropml.models.hierarchical_head import HierarchicalHead  # noqa: E402

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@hydra.main(config_path=str(_CONFIG_DIR / "experiment"), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    backbone_name = str(cfg.get("backbone_name", cfg.backbones[0]))
    seed = int(cfg.get("seed", cfg.seeds[0]))
    checkpoint_path = Path(cfg.checkpoint_dir) / f"{backbone_name}_seed{seed}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Run: python scripts/train.py backbone={backbone_name} seed={seed}"
        )

    print(f"loading checkpoint: {checkpoint_path}")
    backbone = get_backbone(backbone_name)
    head = HierarchicalHead(embedding_dim=backbone.embedding_dim)
    ckpt = torch.load(checkpoint_path, weights_only=True)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])
    backbone.eval(); head.eval()

    export_cfg = ExportConfig(input_size=backbone.config.input_size, opset_version=18)
    onnx_dir = _ARTIFACTS_DIR / f"{backbone_name}_seed{seed}"
    onnx_path = export_to_onnx(backbone, head, onnx_dir / "model.onnx", export_cfg)
    print(f"exported: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.2f} MB)")

    parity = check_parity(onnx_path, backbone, head, export_cfg)
    print(f"parity: crop_diff={parity.crop_logits_max_diff:.2e} "
          f"disease_diff={parity.disease_logits_max_diff:.2e} passed={parity.passed}")
    if not parity.passed:
        raise RuntimeError(f"Parity check FAILED (atol={parity.atol}). Do not deploy this export.")

    manifest = {
        "backbone_name": backbone_name,
        "seed": seed,
        "exported_at": datetime.now(UTC).isoformat(),
        "onnx_path": str(onnx_path.relative_to(_ARTIFACTS_DIR.parent)),
        "weights_sha256": _sha256(onnx_path),
        "input_size": export_cfg.input_size,
        "opset_version": export_cfg.opset_version,
        "n_crops": len(head.crops),
        "n_classes": len(head.classes),
        "crops": list(head.crops),
        "classes": list(head.classes),
        "parity_atol": parity.atol,
        "parity_crop_diff": parity.crop_logits_max_diff,
        "parity_disease_diff": parity.disease_logits_max_diff,
    }
    manifest_path = onnx_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest written: {manifest_path}")


if __name__ == "__main__":
    main()


    