"""Worker entrypoint launched by Accelerate for medical VLM PEFT training."""
import argparse
import gc
import json
import os
import traceback
from pathlib import Path

from foundation_model.config import FoundationTrainConfig
from foundation_model.training.status import write_status
from foundation_model.training.trainer import train_foundation_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = FoundationTrainConfig.from_json_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    status_path = Path(config.output_dir) / "status.json"
    rank = int(os.environ.get("RANK", "0"))
    try:
        if rank == 0:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            write_status(
                status_path,
                state="starting",
                progress=0.0,
                world_size=config.world_size,
                global_batch_size=config.global_batch_size,
                resources_released=False,
            )
        train_foundation_model(config, status_path)
        if rank == 0:
            try:
                current = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                current = {}
            write_status(
                status_path,
                **{
                    **current,
                    "state": "finalizing",
                    "progress": 1.0,
                    "world_size": config.world_size,
                    "global_batch_size": config.global_batch_size,
                    "resources_released": False,
                },
            )
    except Exception as exc:
        if rank == 0:
            write_status(
                status_path,
                state="failed",
                error=str(exc),
                traceback=traceback.format_exc(),
                resources_released=False,
            )
        raise
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
        if rank == 0 and status_path.exists():
            try:
                current = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                current = {}
            if current.get("state") != "failed":
                write_status(
                    status_path,
                    **{**current, "state": "completed", "progress": 1.0, "resources_released": True},
                )
            else:
                write_status(status_path, **{**current, "resources_released": True})


if __name__ == "__main__":
    main()
