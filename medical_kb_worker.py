"""Background worker for medical GraphRAG knowledge-base construction."""
from __future__ import annotations

import argparse
import json
import os
import signal
import traceback
from pathlib import Path

# Medical KB builds must never attempt to fetch an embedding model from Hugging Face.
# The model has to be present on the server and is loaded via local_files_only=True.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from foundation_model.rag.index import GraphRAGIndexManager
from foundation_model.rag.embeddings import validate_local_model_dir


class BuildStopped(Exception):
    pass


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    status_path = Path(cfg["status_path"])
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()

    def stopped(signum, frame):
        raise BuildStopped("用户停止构建")

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stopped)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, stopped)

    latest = {"state": "running", "progress": 0.0, "phase": "starting", "message": "正在准备构建……"}
    _write_json(status_path, latest)

    def on_progress(event: dict):
        nonlocal latest
        latest = {**latest, **event, "state": "running", "output_dir": str(output_dir)}
        _write_json(status_path, latest)

    manager = GraphRAGIndexManager(output_dir)
    try:
        backend = str(cfg.get("embedding_backend") or "local").strip().lower()
        if backend != "local":
            raise ValueError(
                f"后台拒绝在线 Embedding 后端 {backend!r}。医学知识库构建只允许使用本地模型；"
                "请重启 Streamlit 后重新启动构建任务。"
            )
        model_check = validate_local_model_dir(cfg.get("embedding_model"))
        if not model_check.get("valid"):
            raise FileNotFoundError(
                "后台收到的本地 Embedding 模型目录无效，已禁止联网下载。"
                f"路径：{model_check.get('path') or cfg.get('embedding_model')!r}；"
                f"原因：{model_check.get('error')}"
            )
        cfg["embedding_model"] = model_check["path"]
        on_progress({
            "phase": "embedding_check", "progress": 0.44,
            "message": f"已验证本地 Embedding 模型：{model_check['path']}"
        })
        report = manager.build(
            cfg["source_dir"],
            chunk_size=int(cfg.get("chunk_size", 1200)),
            chunk_overlap=int(cfg.get("chunk_overlap", 150)),
            embedding_backend="local",
            embedding_model=cfg["embedding_model"],
            embedding_device=str(cfg.get("embedding_device", "cpu")),
            batch_size=int(cfg.get("batch_size", 16)),
            progress_callback=on_progress,
        )
        validation = manager.validate()
        if not validation.get("valid"):
            raise RuntimeError("知识库写入后校验失败：" + str(validation.get("error") or "unknown"))
        _write_json(status_path, {
            **latest,
            "state": "completed", "progress": 1.0, "phase": "done",
            "message": "医学知识库构建完成。", "report": report.to_dict(),
            "output_dir": str(output_dir),
        })
    except BuildStopped:
        import shutil
        shutil.rmtree(output_dir / ".building", ignore_errors=True)
        _write_json(status_path, {
            **latest, "state": "stopped", "message": "用户已停止构建。",
            "output_dir": str(output_dir),
        })
    except Exception as exc:
        import shutil
        shutil.rmtree(output_dir / ".building", ignore_errors=True)
        _write_json(status_path, {
            **latest, "state": "failed", "message": "构建失败。", "error": str(exc),
            "traceback": traceback.format_exc(), "output_dir": str(output_dir),
        })
        raise


if __name__ == "__main__":
    main()
