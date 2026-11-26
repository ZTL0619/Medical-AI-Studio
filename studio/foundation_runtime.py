"""Process-isolated local Transformers inference from Streamlit."""
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


class TransformersInferenceManager:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()

        self.root = self.project_root / "workspace" / "foundation" / "inference"
        self.root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        base_model,
        image_path,
        prompt,
        adapter_path=None,
        gpu_id=0,
        precision="bf16",
        max_new_tokens=256,
        temperature=0.0,
        trust_remote_code=False,
        history=None,
        quantization="4bit_nf4",
        rag_enabled=False,
        knowledge_base_dir=None,
        rag_top_k=5,
        rag_graph_hops=1,
        rag_context_max_chars=9000,
        rag_visual_max_new_tokens=192,
    ):
        request_id = (
            time.strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:6]
        )

        request_dir = self.root / request_id
        request_dir.mkdir(parents=True, exist_ok=True)

        request_path = request_dir / "request.json"
        output_path = request_dir / "result.json"
        log_path = request_dir / "infer.log"

        payload = {
            "base_model": base_model,
            "image_path": str(image_path),
            "prompt": prompt,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "precision": precision,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "trust_remote_code": bool(trust_remote_code),
            "history": list(history or []),
            "quantization": str(quantization or "none"),
            "rag_enabled": bool(rag_enabled),
            "knowledge_base_dir": str(knowledge_base_dir) if knowledge_base_dir else None,
            "rag_top_k": int(rag_top_k),
            "rag_graph_hops": int(rag_graph_hops),
            "rag_context_max_chars": int(rag_context_max_chars),
            "rag_visual_max_new_tokens": int(rag_visual_max_new_tokens),
        }

        request_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))

        command = [
            sys.executable,
            "foundation_infer_worker.py",
            "--request",
            str(request_path),
            "--output",
            str(output_path),
        ]

        with open(log_path, "wb") as log:
            proc = subprocess.run(
                command,
                cwd=str(self.project_root),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )

        if proc.returncode != 0:
            detail = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            raise RuntimeError(
                f"Transformers inference failed "
                f"(exit={proc.returncode}).\n"
                f"{detail[-4000:]}"
            )

        result = json.loads(
            output_path.read_text(encoding="utf-8")
        )
        result["request_dir"] = str(request_dir)

        return result