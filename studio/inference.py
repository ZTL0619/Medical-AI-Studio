import gc
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ssnet_final.config import build_config
from ssnet_final.engine.checkpoint import load_checkpoint
from ssnet_final.engine.evaluation import infer_la_volume, largest_connected_component
from ssnet_final.engine.metrics import binary_metrics
from ssnet_final.models.factory import build_model
from studio.dataset_tools import load_case


def _algorithm_settings_from_checkpoint(checkpoint):
    """Read saved algorithm settings so UI changes do not silently mismatch a checkpoint."""
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        return {}
    config = payload.get("config")
    if not isinstance(config, dict):
        return {}
    keys = {
        "model",
        "feature_learning",
        "semi_supervision",
        "labeled_percent",
        "anchor_temperature",
        "batch_size",
        "labeled_batch_size",
        "max_iterations",
        "learning_rate",
    }
    return {key: config[key] for key in keys if key in config}


def run_case_inference(draft, checkpoint, case, output_root):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint}")
    output_root = Path(output_root)
    run_dir = output_root / f"inference_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = dict(draft)
    resolved.update(_algorithm_settings_from_checkpoint(checkpoint))
    gpu_ids = [int(v) for v in draft.get("gpu_ids", [0])]
    if torch.cuda.is_available() and draft.get("device", "cuda") != "cpu":
        selected = gpu_ids[0] if gpu_ids else 0
        device = f"cuda:{selected}"
    else:
        device = "cpu"

    data_root = str(draft.get("data_root") or "").strip()
    if not data_root:
        raise ValueError("请先选择医学分割数据根目录。")

    config = build_config(
        command="test",
        dataset="LA",
        model=resolved.get("model", "vnet"),
        feature_learning=resolved.get("feature_learning", "none"),
        semi_supervision=resolved.get("semi_supervision", "none"),
        labeled_percent=resolved.get("labeled_percent", 10),
        data_root=data_root,
        output=run_dir,
        device=device,
        gpu_ids=(gpu_ids[0] if gpu_ids else 0,),
        parallel_mode="single",
        checkpoint=checkpoint,
        anchor_temperature=resolved.get("anchor_temperature", 100.0),
        batch_size=resolved.get("batch_size"),
        labeled_batch_size=resolved.get("labeled_batch_size"),
        max_iterations=resolved.get("max_iterations"),
        learning_rate=resolved.get("learning_rate"),
    )
    model = build_model(config, training=False).to(torch.device(device))
    ckpt_iteration, ckpt_best_score, load_result = load_checkpoint(
        checkpoint, model, map_location=device, expected_config=config
    )
    model.eval()
    image, label = load_case(config.data_root, case)
    # CCAI test_LA.py uses test_3d_patch_3 with --nms=1 by default: decoder-1
    # softmax + sliding-window averaging + argmax + largest connected component.
    raw_prediction, score = infer_la_volume(config, model, image, 18, 4)
    raw_metrics = binary_metrics(raw_prediction.astype(bool), label.astype(bool))
    prediction = largest_connected_component(raw_prediction)
    metrics = binary_metrics(prediction.astype(bool), label.astype(bool))

    pred_path = run_dir / f"{case}_pred.nii.gz"
    score_path = run_dir / f"{case}_scores.nii.gz"
    nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), pred_path)
    # The CCAI test code saves the complete [C, W, H, D] score map rather
    # than silently discarding the background channel.
    nib.save(nib.Nifti1Image(score.astype(np.float32), np.eye(4)), score_path)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "case": case,
                "metrics": metrics,
                "raw_metrics": raw_metrics,
                "checkpoint": {
                    "path": str(checkpoint),
                    "iteration": ckpt_iteration,
                    "best_score": ckpt_best_score,
                    "missing_keys": list(load_result.missing_keys),
                    "unexpected_keys": list(load_result.unexpected_keys),
                },
                "algorithm": {
                    "model": config.model,
                    "feature_learning": config.feature_learning,
                    "semi_supervision": config.semi_supervision,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    result = {
        "case": case,
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "prediction_path": str(pred_path),
        "score_path": str(score_path),
        "output_dir": str(run_dir),
        "prediction": prediction,
        "image": image,
        "label": label,
        "checkpoint": {
            "path": str(checkpoint),
            "iteration": ckpt_iteration,
            "best_score": ckpt_best_score,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
        "algorithm": {
            "model": config.model,
            "feature_learning": config.feature_learning,
            "semi_supervision": config.semi_supervision,
        },
    }
    # Inference runs inside the long-lived Streamlit process.  PyTorch's CUDA
    # caching allocator would otherwise keep released blocks reserved and steal
    # memory from the next training subprocess.
    model = None
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    return result
