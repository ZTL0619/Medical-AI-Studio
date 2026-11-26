"""Lifecycle manager for an OpenAI-compatible vLLM medical VLM service."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

import psutil

from foundation_model.config import FoundationServingConfig


class VLLMServiceManager:
    def __init__(self, project_root):
        self.root = Path(project_root).resolve() / "workspace" / "foundation" / "serving"
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "service.json"
        self.log_path = self.root / "vllm.log"

    @staticmethod
    def _alive(pid):
        try:
            p = psutil.Process(int(pid))
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    def _read(self):
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write(self, payload):
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.meta_path)

    def build_command(self, config: FoundationServingConfig):
        config.validate()
        executable = shutil.which("vllm")
        if not executable:
            raise RuntimeError("vLLM executable not found. Install the project 'serving' extra first.")
        command = [
            executable, "serve", config.base_model,
            "--host", config.host,
            "--port", str(config.port),
            "--served-model-name", config.served_model_name,
            "--dtype", "bfloat16" if config.precision == "bf16" else "bfloat16",
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        ]
        if config.strategy == "data_parallel" and len(config.gpu_ids) > 1:
            command += ["--data-parallel-size", str(len(config.gpu_ids))]
        elif config.strategy == "tensor_parallel" and len(config.gpu_ids) > 1:
            command += ["--tensor-parallel-size", str(len(config.gpu_ids))]
        if config.precision == "int4":
            command += ["--quantization", "bitsandbytes"]
        if config.max_model_len:
            command += ["--max-model-len", str(int(config.max_model_len))]
        if config.trust_remote_code:
            command += ["--trust-remote-code"]
        if config.adapter_path:
            command += [
                "--enable-lora",
                "--lora-modules", f"{config.adapter_name}={config.adapter_path}",
            ]
        return command

    def start(self, config: FoundationServingConfig):
        current = self._read()
        if current and self._alive(current.get("pid")):
            raise RuntimeError(f"vLLM service already running, PID={current['pid']}")
        command = self.build_command(config)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in config.gpu_ids)
        log = open(self.log_path, "ab", buffering=0)
        kwargs = {"cwd": str(self.root.parent.parent.parent), "env": env, "stdout": log, "stderr": subprocess.STDOUT}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(command, **kwargs)
        finally:
            log.close()
        meta = {
            "pid": proc.pid,
            "created_at": time.time(),
            "command": command,
            "config": config.to_json_dict(),
            "endpoint": config.endpoint,
            "model_for_request": config.adapter_name if config.adapter_path else config.served_model_name,
            "log_path": str(self.log_path),
        }
        self._write(meta)
        return meta

    def healthy(self, timeout=1.0):
        meta = self._read()
        if not meta or not self._alive(meta.get("pid")):
            return False
        try:
            with urlopen(meta["endpoint"].removesuffix("/v1") + "/health", timeout=timeout) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def status(self):
        meta = self._read()
        if not meta:
            return {"state": "stopped"}
        alive = self._alive(meta.get("pid"))
        return {**meta, "state": "running" if alive else "stopped", "healthy": self.healthy() if alive else False}

    def stop(self):
        meta = self._read()
        if not meta or not self._alive(meta.get("pid")):
            return {"stopped": False, "message": "vLLM service is not running."}
        root = psutil.Process(int(meta["pid"]))
        procs = root.children(recursive=True) + [root]
        for proc in procs:
            try:
                proc.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(procs, timeout=10)
        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass
        return {"stopped": True}

    def tail_log(self, lines=100):
        if not self.log_path.exists():
            return ""
        return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-int(lines):])
