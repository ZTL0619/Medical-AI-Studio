"""Experiment provenance helpers.

Each training job writes a machine-readable manifest next to the checkpoints so
an experiment can be audited/reproduced without relying on UI state.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def _git_commit(project_root: Path):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def build_manifest(config, project_root):
    project_root = Path(project_root)
    return {
        "schema_version": 1,
        "experiment": config.to_json_dict(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        },
        "source": {
            "git_commit": _git_commit(project_root),
        },
    }


def write_manifest(output_dir, config, project_root):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "reproducibility.json"
    path.write_text(
        json.dumps(build_manifest(config, project_root), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
