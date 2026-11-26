"""Checkpoint schema and compatibility helpers.

Checkpoints are saved atomically and carry enough model/training metadata for
inference to reject obvious architecture mismatches instead of silently loading
an incompatible state dict.
"""

import os
import tempfile
from pathlib import Path

import torch


CHECKPOINT_FORMAT_VERSION = 2


def _config_payload(config):
    return config.to_json_dict() if hasattr(config, "to_json_dict") else dict(config.__dict__)


def _atomic_torch_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def save_checkpoint(path, model, optimizer, iteration, best_score, config):
    path = Path(path)
    config_payload = _config_payload(config)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "iteration": int(iteration),
        "best_score": float(best_score),
        "model_meta": {
            "model": config_payload.get("model"),
            "num_classes": config_payload.get("num_classes"),
            "patch_size": config_payload.get("patch_size"),
            "feature_learning": config_payload.get("feature_learning"),
            "semi_supervision": config_payload.get("semi_supervision"),
        },
        "config": config_payload,
    }
    _atomic_torch_save(payload, path)


def _adapt_legacy_state_dict(state, model):
    """Map older standalone model checkpoints into the modular wrapper."""
    state = {key.removeprefix("module."): value for key, value in state.items()}
    current = model.state_dict()
    if any(key.startswith("backbone.") for key in state):
        return state

    adapted = {}
    dnp_prefixes = (
        "latent_path.",
        "deterministic_path.",
        "mean_net.",
        "log_var_net.",
        "fc_decoder.",
        "classifier.",
        "projection_head.",
        "prediction_head.",
        "contrastive_class_selector_",
        "contrastive_class_selector_memory",
    )
    for key, value in state.items():
        candidates = []
        if key.startswith(dnp_prefixes):
            candidates.append("dnp_head." + key)
        if key == "projection_head.weight":
            candidates.append("feature_learning.transform.weight")
        if key == "projection_head.bias":
            candidates.append("feature_learning.transform.bias")
        if key.startswith("projection_head.0."):
            candidates.append("feature_learning.transform." + key.split("projection_head.0.", 1)[1])
        candidates.append("backbone." + key)
        candidates.append(key)
        for candidate in candidates:
            if candidate in current and current[candidate].shape == value.shape:
                adapted[candidate] = value
                break
    return adapted


def validate_checkpoint_compatibility(payload, expected_config):
    """Reject model-level mismatches while remaining backward compatible."""
    if expected_config is None or not isinstance(payload, dict):
        return
    saved = payload.get("model_meta") or payload.get("config") or {}
    checks = {
        "model": getattr(expected_config, "model", None),
        "num_classes": getattr(expected_config, "num_classes", None),
    }
    for key, expected in checks.items():
        actual = saved.get(key)
        if actual is not None and expected is not None and actual != expected:
            raise RuntimeError(
                f"Checkpoint compatibility error: {key}={actual!r}, expected {expected!r}."
            )
    saved_patch = saved.get("patch_size")
    expected_patch = getattr(expected_config, "patch_size", None)
    if saved_patch is not None and expected_patch is not None:
        if tuple(saved_patch) != tuple(expected_patch):
            raise RuntimeError(
                f"Checkpoint compatibility error: patch_size={tuple(saved_patch)}, "
                f"expected {tuple(expected_patch)}."
            )


def load_checkpoint(path, model, optimizer=None, map_location="cpu", expected_config=None):
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "model" in payload:
        state = payload["model"]
    else:
        state = payload
        payload = {"model": state, "iteration": 0, "best_score": 0.0, "optimizer": None}

    validate_checkpoint_compatibility(payload, expected_config)
    state = _adapt_legacy_state_dict(state, model)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        # Contrastive heads were added to reproduce the selected aux4 training
        # loss. They are not used during segmentation inference, so older
        # checkpoints may safely omit only these auxiliary parameters.
        allowed_missing_prefixes = (
            "dnp_head.projection_head.",
            "dnp_head.prediction_head.",
            "dnp_head.contrastive_class_selector_",
            "dnp_head.contrastive_class_selector_memory",
        )
        required_missing = [
            key for key in result.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if required_missing:
            raise RuntimeError(f"Checkpoint is missing model parameters: {required_missing}")
    if result.unexpected_keys:
        raise RuntimeError(f"Checkpoint has unexpected model parameters: {list(result.unexpected_keys)}")
    if optimizer is not None and payload.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except ValueError:
            pass
    return int(payload.get("iteration", 0)), float(payload.get("best_score", 0.0)), result
