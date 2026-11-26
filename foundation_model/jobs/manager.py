"""Process-isolated job manager for medical VLM PEFT training."""
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import psutil

from foundation_model.config import FoundationTrainConfig
from foundation_model.data.jsonl_dataset import validate_jsonl_dataset
from foundation_model.distributed.launcher import accelerate_launch_command, launch_environment
from foundation_model.runtime.dependencies import require


class FoundationJobManager:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.workspace = self.project_root / "workspace" / "foundation"
        self.jobs_root = self.workspace / "jobs"
        self.experiments_root = self.workspace / "experiments"
        self.latest_path = self.workspace / "latest_job.json"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.experiments_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _alive(pid):
        try:
            proc = psutil.Process(int(pid))
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
            return False

    @staticmethod
    def _write_json(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def latest_job(self):
        try:
            meta = json.loads(self.latest_path.read_text(encoding="utf-8"))
            return meta
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _assert_released(self):
        meta = self.latest_job()
        if meta and self._alive(meta.get("pid")):
            raise RuntimeError(
                f"上一大模型训练任务仍在运行/释放资源（PID={meta.get('pid')}）。"
                "请等待进程完全退出后再启动下一次训练。"
            )

    def create_config(self, draft, job_id=None):
        job_id = job_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        name = str(draft.get("experiment_name") or "medvlm_qlora")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_") or "medvlm"
        output = self.experiments_root / f"{safe}_{job_id}"
        config = FoundationTrainConfig(
            experiment_name=name,
            base_model=str(draft.get("base_model") or "google/medgemma-4b-it"),
            dataset_path=Path(draft.get("dataset_path") or "examples/foundation_dataset_sample.jsonl").expanduser(),
            output_dir=output,
            gpu_ids=tuple(int(x) for x in draft.get("gpu_ids", [0, 1])),
            distributed_strategy=str(draft.get("distributed_strategy", "ddp")),
            peft_strategy=str(draft.get("peft_strategy", "qlora")),
            precision=str(draft.get("train_precision", "bf16")),
            per_device_train_batch_size=int(draft.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(draft.get("gradient_accumulation_steps", 8)),
            num_train_epochs=float(draft.get("num_train_epochs", 1.0)),
            learning_rate=float(draft.get("learning_rate", 2e-5)),
            warmup_ratio=float(draft.get("warmup_ratio", 0.03)),
            weight_decay=float(draft.get("weight_decay", 0.0)),
            logging_steps=int(draft.get("logging_steps", 5)),
            save_steps=int(draft.get("save_steps", 100)),
            seed=int(draft.get("seed", 1337)),
            gradient_checkpointing=bool(draft.get("gradient_checkpointing", True)),
            lora_rank=int(draft.get("lora_rank", 16)),
            lora_alpha=float(draft.get("lora_alpha", 16.0)),
            lora_dropout=float(draft.get("lora_dropout", 0.05)),
            lora_target_modules=str(draft.get("lora_target_modules", "all-linear")),
            trust_remote_code=bool(draft.get("trust_remote_code", False)),
            resume_from_checkpoint=Path(draft["resume_from_checkpoint"]).expanduser() if draft.get("resume_from_checkpoint") else None,
            validation_dataset_path=Path(draft["validation_dataset_path"]).expanduser() if draft.get("validation_dataset_path") else None,
            validation_split=float(draft.get("validation_split", 0.1)),
        ).validate()
        return job_id, config

    def start(self, draft):
        self._assert_released()
        require(["accelerate"])
        job_id, config = self.create_config(draft)
        report = validate_jsonl_dataset(config.dataset_path, check_images=True)
        if not report["valid"]:
            raise FileNotFoundError("SFT dataset references missing images: " + ", ".join(report["missing_images"][:5]))
        if config.validation_dataset_path:
            val_report = validate_jsonl_dataset(config.validation_dataset_path, check_images=True)
            if not val_report["valid"]:
                raise FileNotFoundError(
                    "Validation dataset references missing images: "
                    + ", ".join(val_report["missing_images"][:5])
                )

        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        config_path = job_dir / "config.json"
        self._write_json(config_path, config.to_json_dict())
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "resolved_config.json").write_text(
            json.dumps(config.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log_path = job_dir / "train.log"
        command = accelerate_launch_command(config, config_path)
        env = launch_environment(config)
        log_handle = open(log_path, "ab", buffering=0)
        kwargs = {
            "cwd": str(self.project_root), "env": env,
            "stdout": log_handle, "stderr": subprocess.STDOUT,
        }
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
            "output": str(config.output_dir),
            "status_path": str(config.output_dir / "status.json"),
            "log_path": str(log_path),
            "config_path": str(config_path),
            "command": command,
        }
        self._write_json(job_dir / "job.json", meta)
        self._write_json(self.latest_path, meta)
        return meta

    def status(self):
        meta = self.latest_job()
        if not meta:
            return {"state": "idle"}
        try:
            state = json.loads(Path(meta["status_path"]).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            state = {"state": "starting"}
        alive = self._alive(meta.get("pid"))
        state.update({"pid": meta.get("pid"), "process_alive": alive, "output": meta.get("output"), "job_id": meta.get("job_id")})
        if alive and state.get("state") == "completed":
            state["state"] = "finalizing"
        if not alive and state.get("state") in {"starting", "running", "finalizing"}:
            state["state"] = "completed" if state.get("resources_released") else "stopped"
        return state

    def stop(self):
        meta = self.latest_job()
        if not meta or not self._alive(meta.get("pid")):
            return {"stopped": False, "message": "没有正在运行的大模型训练任务。"}
        root = psutil.Process(int(meta["pid"]))
        procs = root.children(recursive=True) + [root]
        for proc in procs:
            try:
                proc.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(procs, timeout=8)
        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass
        return {"stopped": True, "job_id": meta["job_id"]}

    def tail_log(self, lines=100):
        meta = self.latest_job()
        if not meta:
            return ""
        path = Path(meta["log_path"])
        if not path.exists():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-int(lines):])
