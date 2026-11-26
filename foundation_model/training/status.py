"""Atomic status writer for foundation-model jobs."""
import json
import os
import tempfile
import time
from pathlib import Path


def write_status(path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "updated_at": time.time()}
    fd, tmp = tempfile.mkstemp(prefix="status_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
