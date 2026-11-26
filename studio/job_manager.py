import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil
import torch

from ssnet_final.config import build_config
from ssnet_final.data.ua_mt import layout_status, resolve_data_root
from studio.reproducibility import write_manifest
from studio.experiment_db import ExperimentDB


class JobManager:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.workspace = self.project_root / "workspace"
        self.jobs_root = self.workspace / "jobs"
        self.experiments_root = self.workspace / "experiments"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.experiments_root.mkdir(parents=True, exist_ok=True)
        self.db = ExperimentDB(self.workspace / "experiments.sqlite3")

    @staticmethod
    def _process_alive(pid):
        """Return True only for a live, non-zombie worker process."""
        try:
            proc = psutil.Process(int(pid))
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
            return False

    def _assert_previous_job_released(self):
        """Prevent overlapping training workers on the same server/GPU."""
        previous = self.latest_job()
        if previous is None:
            return
        if self._process_alive(previous.get("pid")):
            try:
                state = json.loads(Path(previous["status_path"]).read_text(encoding="utf-8")).get("state", "unknown")
            except Exception:
                state = "unknown"
            raise RuntimeError(
                f"上一个训练任务进程仍未退出（PID={previous['pid']}, state={state}）。"
                "请等待界面状态从 finalizing 变为 completed 后再开始新训练。"
            )

    def _job_meta_path(self, job_id):
        return self.jobs_root / job_id / "job.json"

    def _write_json(self, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def create_config(self, draft, job_id=None):
        job_id = job_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        experiment_name = str(draft.get("experiment_name", "")).strip()
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in experiment_name).strip("_")
        auto_name = (
            f"{draft['model']}_feat-{draft['feature_learning']}_"
            f"semi-{draft['semi_supervision']}_la_{draft['labeled_percent']}pct"
        )
        output_name = f"{safe_name or auto_name}_{job_id}"
        output = self.experiments_root / output_name
        gpu_ids = tuple(int(v) for v in draft.get("gpu_ids", [0]))
        parallel_mode = draft.get("parallel_mode", "single")
        if parallel_mode == "ddp" and len(gpu_ids) < 2:
            raise ValueError("多卡 DDP 至少需要选择两张 GPU。")
        if str(draft.get("device", "cuda")).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("当前环境没有可用 CUDA GPU。")

        data_root = str(draft.get("data_root") or "").strip()
        if not data_root:
            raise ValueError("请先选择医学分割数据根目录。")
        device = "cpu" if draft.get("device") == "cpu" else "cuda:0"
        config = build_config(
            command="train",
            dataset=draft.get("dataset", "LA"),
            experiment_name=draft.get("experiment_name", ""),
            model=draft.get("model", "vnet"),
            feature_learning=draft.get("feature_learning", "none"),
            semi_supervision=draft.get("semi_supervision", "none"),
            labeled_percent=draft.get("labeled_percent", 10),
            data_root=resolve_data_root(data_root),
            output=output,
            device=device,
            seed=draft.get("seed", 1337),
            workers=draft.get("workers", 4),
            validation_interval=draft.get("validation_interval", 200),
            save_interval=draft.get("save_interval", 1000),
            log_interval=draft.get("log_interval", 10),
            anchor_temperature=draft.get("anchor_temperature", 100.0),
            parallel_mode=parallel_mode,
            gpu_ids=gpu_ids,
            mixed_precision=draft.get("mixed_precision", False),
            sync_batchnorm=draft.get("sync_batchnorm", False),
            linear_lr_scaling=draft.get("linear_lr_scaling", False),
            ddp_find_unused_parameters=draft.get("ddp_find_unused_parameters", False),
            batch_size=draft.get("batch_size"),
            labeled_batch_size=draft.get("labeled_batch_size"),
            max_iterations=draft.get("max_iterations"),
            learning_rate=draft.get("learning_rate"),
        )
        return job_id, config

    def start_training(self, draft):
        self._assert_previous_job_released()
        job_id, config = self.create_config(draft)
        data_root = Path(config.data_root)
        status = layout_status(data_root)
        if not status["train_list_exists"]:
            raise FileNotFoundError(f"未找到 {status['train_list']}")
        if not status["test_list_exists"]:
            raise FileNotFoundError(f"未找到 {status['test_list']}")
        if not status["case_dir_exists"]:
            raise FileNotFoundError(f"未找到 UA-MT 病例目录: {status['case_dir']}")

        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        launch_config = job_dir / "launch_config.json"
        self._write_json(launch_config, config.to_json_dict())
        write_manifest(config.output, config, self.project_root)
        log_path = job_dir / "train.log"
        env = os.environ.copy()

        if config.device.startswith("cuda"):
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(v) for v in config.gpu_ids)

        if config.parallel_mode == "ddp":
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={len(config.gpu_ids)}",
                "train_worker.py",
                "--config",
                str(launch_config),
            ]
        else:
            command = [sys.executable, "train_worker.py", "--config", str(launch_config)]

        log_handle = open(log_path, "ab", buffering=0)
        popen_kwargs = {
            "cwd": str(self.project_root),
            "env": env,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        try:
            process = subprocess.Popen(command, **popen_kwargs)
        finally:
            # The child inherited the descriptor; the Streamlit parent must not
            # keep one open for every training run.
            log_handle.close()
        meta = {
            "job_id": job_id,
            "pid": process.pid,
            "created_at": time.time(),
            "command": command,
            "output": str(config.output),
            "status_path": str(config.output / "status.json"),
            "log_path": str(log_path),
            "config_path": str(launch_config),
        }
        self._write_json(self._job_meta_path(job_id), meta)
        self.db.upsert(
            job_id,
            "starting",
            config.output,
            config.to_json_dict(),
            pid=process.pid,
            command=command,
        )
        return meta

    def load_job(self, job_id):
        path = self._job_meta_path(job_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown job: {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def latest_job(self):
        rows = self.db.list_recent(1)
        if not rows:
            return None
        return self.load_job(rows[0]["job_id"])

    def get_status(self, job_id=None):
        meta = self.latest_job() if job_id is None else self.load_job(job_id)
        if meta is None:
            return {"state": "idle"}
        status_path = Path(meta["status_path"])
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            status = {"state": "starting"}
        status["job_id"] = meta["job_id"]
        status["output"] = meta["output"]
        status["log_path"] = meta["log_path"]
        status["pid"] = meta["pid"]
        status["process_alive"] = self._process_alive(meta["pid"])
        if status["process_alive"] and status.get("state") == "completed":
            # A terminal status file can be written milliseconds before the OS
            # has torn down the CUDA context.  Keep the UI locked until PID exit.
            status["state"] = "finalizing"
        if not status["process_alive"] and status.get("state") in {"starting", "running", "finalizing"}:
            if status.get("resources_released"):
                status["state"] = "completed"
            elif status.get("state") != "finalizing":
                status["state"] = "stopped"
        self.db.update_state(meta["job_id"], status["state"])
        return status

    def stop_training(self, job_id=None):
        meta = self.latest_job() if job_id is None else self.load_job(job_id)
        if meta is None:
            return {"stopped": False, "message": "没有正在记录的训练任务。"}
        pid = int(meta["pid"])
        if not psutil.pid_exists(pid):
            return {"stopped": False, "message": "训练进程已经结束。"}
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for proc in children:
            try:
                proc.terminate()
            except psutil.Error:
                pass
        try:
            parent.terminate()
        except psutil.Error:
            pass
        _, alive = psutil.wait_procs(children + [parent], timeout=5)
        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass
        status_path = Path(meta["status_path"])
        current = {}
        try:
            current = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        current.update({"state": "stopped", "updated_at": time.time()})
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        self.db.update_state(meta["job_id"], "stopped")
        return {"stopped": True, "job_id": meta["job_id"]}

    def tail_log(self, job_id=None, lines=80):
        meta = self.latest_job() if job_id is None else self.load_job(job_id)
        if meta is None:
            return ""
        path = Path(meta["log_path"])
        if not path.exists():
            return ""
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(data[-int(lines) :])
