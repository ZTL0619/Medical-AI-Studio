"""Process-isolated medical knowledge-base build jobs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil


class MedicalKBJobManager:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.workspace = self.project_root / "workspace" / "foundation" / "kb_jobs"
        self.latest_path = self.project_root / "workspace" / "foundation" / "latest_kb_job.json"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _alive(pid):
        try:
            proc = psutil.Process(int(pid))
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
            return False

    def latest_job(self):
        try:
            return json.loads(self.latest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def status(self):
        meta = self.latest_job()
        if not meta:
            return {"state": "idle", "process_alive": False}
        try:
            state = json.loads(Path(meta["status_path"]).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            state = {"state": "starting", "progress": 0.0}
        alive = self._alive(meta.get("pid"))
        state.update({
            "pid": meta.get("pid"),
            "job_id": meta.get("job_id"),
            "process_alive": alive,
            "source_dir": meta.get("source_dir"),
            "output_dir": meta.get("output_dir"),
            "log_path": meta.get("log_path"),
        })
        if not alive and state.get("state") in {"starting", "running"}:
            state["state"] = "stopped"
            state.setdefault("message", "构建进程已停止。")
        return state

    def start(self, config: dict):
        current = self.status()
        if current.get("process_alive"):
            raise RuntimeError(f"已有医学知识库构建任务正在运行（PID={current.get('pid')}）。")

        # This builder is intentionally offline-only. Never allow a stale Streamlit
        # process/config to start an hf/modelscope download job.
        from foundation_model.rag.embeddings import validate_local_model_dir
        backend = str(config.get("embedding_backend") or "local").strip().lower()
        if backend != "local":
            raise ValueError(
                f"建库只允许使用服务器本地 Embedding 模型，但当前界面传入 backend={backend!r}。"
                "请停止旧版 Streamlit 并重新启动 app.py，再重新选择本地 BGE 模型目录。"
            )
        model_check = validate_local_model_dir(config.get("embedding_model"))
        if not model_check.get("valid"):
            raise FileNotFoundError(
                "本地 Embedding 模型路径无效，已阻止后台联网下载。"
                f"路径：{model_check.get('path') or config.get('embedding_model')!r}；"
                f"原因：{model_check.get('error')}"
            )
        config = {**config, "embedding_backend": "local", "embedding_model": model_check["path"]}

        job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        job_dir = self.workspace / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        status_path = job_dir / "status.json"
        config_path = job_dir / "config.json"
        log_path = job_dir / "build.log"
        payload = dict(config)
        payload["status_path"] = str(status_path)
        self._write_json(config_path, payload)
        self._write_json(status_path, {
            "state": "starting", "progress": 0.0, "phase": "starting",
            "message": "正在启动医学知识库构建任务……",
        })
        command = [sys.executable, str(self.project_root / "medical_kb_worker.py"), "--config", str(config_path)]
        log_handle = open(log_path, "ab", buffering=0)
        kwargs = {"cwd": str(self.project_root), "stdout": log_handle, "stderr": subprocess.STDOUT}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen(command, **kwargs)
        finally:
            log_handle.close()
        meta = {
            "job_id": job_id,
            "pid": process.pid,
            "created_at": time.time(),
            "status_path": str(status_path),
            "config_path": str(config_path),
            "log_path": str(log_path),
            "source_dir": str(payload.get("source_dir") or ""),
            "output_dir": str(payload.get("output_dir") or ""),
            "command": command,
        }
        self._write_json(job_dir / "job.json", meta)
        self._write_json(self.latest_path, meta)
        return meta

    def stop(self):
        meta = self.latest_job()
        if not meta or not self._alive(meta.get("pid")):
            return {"stopped": False, "message": "没有正在运行的医学知识库构建任务。"}
        root = psutil.Process(int(meta["pid"]))
        procs = root.children(recursive=True) + [root]
        for proc in procs:
            try:
                proc.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(procs, timeout=5)
        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass
        try:
            status = json.loads(Path(meta["status_path"]).read_text(encoding="utf-8"))
        except Exception:
            status = {}
        status.update({"state": "stopped", "message": "用户已停止构建。", "stopped_at": time.time()})
        self._write_json(meta["status_path"], status)
        output_dir = Path(meta.get("output_dir") or "")
        if output_dir:
            import shutil
            shutil.rmtree(output_dir / ".building", ignore_errors=True)
        return {"stopped": True, "job_id": meta.get("job_id")}

    def tail_log(self, lines=80):
        meta = self.latest_job()
        if not meta:
            return ""
        path = Path(meta.get("log_path") or "")
        if not path.exists():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-int(lines):])
