"""DeepSpeed ZeRO-2 configuration for the optional enterprise training path."""
import json
from pathlib import Path


def zero2_config(config):
    precision_key = "bf16" if config.precision == "bf16" else "fp16"
    payload = {
        "train_micro_batch_size_per_gpu": int(config.per_device_train_batch_size),
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        precision_key: {"enabled": True},
        "gradient_clipping": 1.0,
        "steps_per_print": 1000000,
    }
    return payload


def write_zero2_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(zero2_config(config), indent=2), encoding="utf-8")
    return path
