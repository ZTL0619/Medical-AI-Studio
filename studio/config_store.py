import json
import os
import tempfile
from pathlib import Path


DEFAULT_DRAFT = {
    "dataset": "LA",
    "experiment_name": "",
    "data_root": "",
    "device": "cuda",
    "model": "vnet",
    "feature_learning": "none",
    "semi_supervision": "none",
    "labeled_percent": 10,
    "seed": 1337,
    "workers": 4,
    "validation_interval": 200,
    "save_interval": 1000,
    "log_interval": 10,
    "anchor_temperature": 100.0,
    "parallel_mode": "single",
    "gpu_ids": [0],
    "mixed_precision": False,
    "sync_batchnorm": False,
    "linear_lr_scaling": False,
    "ddp_find_unused_parameters": False,
    "batch_size": None,
    "labeled_batch_size": None,
    "max_iterations": None,
    "learning_rate": None,
    "resume": None,
}



class ConfigStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self):
        values = dict(DEFAULT_DRAFT)
        try:
            values.update(json.loads(self.path.read_text(encoding="utf-8")))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return values

    def write(self, values):
        payload = {**DEFAULT_DRAFT, **values}
        fd, tmp = tempfile.mkstemp(prefix="config_", suffix=".json", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            Path(tmp).replace(self.path)
        finally:
            if Path(tmp).exists():
                Path(tmp).unlink(missing_ok=True)
        return payload

    def update(self, patch):
        values = self.read()
        values.update({key: value for key, value in patch.items() if value is not None})
        return self.write(values)
