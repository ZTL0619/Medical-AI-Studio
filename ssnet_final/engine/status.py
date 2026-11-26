import json
import os
import tempfile
import time
from pathlib import Path


class StatusWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **payload):
        payload = {"updated_at": time.time(), **payload}
        self._atomic_json(payload)

    def update(self, **changes):
        current = self.read()
        current.update(changes)
        current["updated_at"] = time.time()
        self._atomic_json(current)

    def read(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _atomic_json(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
