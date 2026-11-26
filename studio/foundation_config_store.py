"""Persistent UI draft for the medical foundation-model workspace."""
import json
import os
import tempfile
from pathlib import Path

from foundation_model.rag.defaults import default_local_embedding_model


DEFAULT_FOUNDATION_DRAFT = {
    "experiment_name": "medvlm_qlora",
    "model_key": "medgemma4b",
    "base_model": "google/medgemma-4b-it",
    "dataset_path": "examples/foundation_dataset_sample.jsonl",
    "validation_dataset_path": None,
    "validation_split": 0.1,
    "gpu_ids": [0],
    "distributed_strategy": "ddp",
    "peft_strategy": "qlora",
    "train_precision": "bf16",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 1.0,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "logging_steps": 5,
    "save_steps": 100,
    "seed": 1337,
    "gradient_checkpointing": True,
    "lora_rank": 16,
    "lora_alpha": 16.0,
    "lora_dropout": 0.05,
    "lora_target_modules": "all-linear",
    "resume_from_checkpoint": None,
    "trust_remote_code": False,
    "adapter_path": None,
    "inference_backend": "vllm",
    "inference_precision": "bf16",
    "inference_quantization": "4bit_nf4",
    "finetune_enabled": False,
    "serving_strategy": "data_parallel",
    "serving_precision": "bf16",
    "serving_host": "127.0.0.1",
    "serving_port": 8010,
    "served_model_name": "medical-vlm",
    "adapter_name": "medical-adapter",
    "gpu_memory_utilization": 0.88,
    "max_model_len": None,
    "max_new_tokens": 256,
    "temperature": 0.0,
    "rag_enabled": False,
    "rag_top_k": 5,
    "rag_graph_hops": 1,
    "rag_embedding_backend": "local",
    "rag_embedding_model": default_local_embedding_model(),
    "rag_chunk_size": 1200,
    "rag_chunk_overlap": 150,
    "rag_context_max_chars": 9000,
    "rag_visual_max_new_tokens": 192,
    "knowledge_source_dir": None,
    "knowledge_base_dir": None,
}


class FoundationConfigStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self):
        values = dict(DEFAULT_FOUNDATION_DRAFT)
        try:
            values.update(json.loads(self.path.read_text(encoding="utf-8")))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return values

    def write(self, values):
        payload = {**DEFAULT_FOUNDATION_DRAFT, **values}
        fd, tmp = tempfile.mkstemp(prefix="foundation_config_", suffix=".json", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        finally:
            if Path(tmp).exists():
                Path(tmp).unlink(missing_ok=True)
        return payload
