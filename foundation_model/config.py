"""Serializable configuration contracts for the medical VLM engineering branch.

This module intentionally contains no Transformers/vLLM imports so the core 3-D
segmentation application remains usable without installing the optional LLM stack.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from foundation_model.rag.defaults import default_local_embedding_model


@dataclass(frozen=True)
class FoundationTrainConfig:
    experiment_name: str = "medvlm_qlora"
    base_model: str = "google/medgemma-4b-it"
    dataset_path: Path = Path("examples/foundation_dataset_sample.jsonl")
    output_dir: Path = Path("workspace/foundation_experiments/medvlm_qlora")
    gpu_ids: Tuple[int, ...] = (0,)
    distributed_strategy: str = "ddp"  # ddp | deepspeed_zero2
    peft_strategy: str = "qlora"  # qlora | lora
    precision: str = "bf16"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 5
    save_steps: int = 100
    seed: int = 1337
    gradient_checkpointing: bool = True
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    trust_remote_code: bool = False
    resume_from_checkpoint: Optional[Path] = None
    validation_dataset_path: Optional[Path] = None
    validation_split: float = 0.1
    auto_evaluate: bool = True
    evaluation_max_samples: int = 20
    evaluation_max_new_tokens: int = 128

    @property
    def world_size(self) -> int:
        return len(self.gpu_ids)

    @property
    def global_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.world_size * self.gradient_accumulation_steps

    def validate(self):
        if not self.base_model:
            raise ValueError("base_model cannot be empty")
        if len(self.gpu_ids) < 1:
            raise ValueError("At least one GPU must be selected")
        if self.distributed_strategy not in {"ddp", "deepspeed_zero2"}:
            raise ValueError("distributed_strategy must be ddp or deepspeed_zero2")
        if self.peft_strategy not in {"qlora", "lora"}:
            raise ValueError("peft_strategy must be qlora or lora")
        if self.precision not in {"bf16", "fp16"}:
            raise ValueError("precision must be bf16 or fp16")
        if self.per_device_train_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch size and gradient accumulation must be positive")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank/alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0,1)")
        if not 0 < float(self.validation_split) < 1:
            raise ValueError("validation_split must be in (0,1)")
        if int(self.evaluation_max_samples) <= 0:
            raise ValueError("evaluation_max_samples must be positive")
        if int(self.evaluation_max_new_tokens) <= 0:
            raise ValueError("evaluation_max_new_tokens must be positive")
        return self

    def to_json_dict(self):
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
            elif isinstance(value, tuple):
                payload[key] = list(value)
        return payload

    @classmethod
    def from_json_dict(cls, values):
        payload = dict(values)
        payload["dataset_path"] = Path(payload["dataset_path"])
        payload["output_dir"] = Path(payload["output_dir"])
        payload["gpu_ids"] = tuple(int(v) for v in payload.get("gpu_ids", [0]))
        if payload.get("resume_from_checkpoint"):
            payload["resume_from_checkpoint"] = Path(payload["resume_from_checkpoint"])
        if payload.get("validation_dataset_path"):
            payload["validation_dataset_path"] = Path(payload["validation_dataset_path"])
        return cls(**payload).validate()


@dataclass(frozen=True)
class FoundationServingConfig:
    base_model: str = "google/medgemma-4b-it"
    adapter_path: Optional[Path] = None
    gpu_ids: Tuple[int, ...] = (0,)
    strategy: str = "data_parallel"  # data_parallel | tensor_parallel
    precision: str = "bf16"  # bf16 | int4
    host: str = "127.0.0.1"
    port: int = 8010
    served_model_name: str = "medical-vlm"
    adapter_name: str = "medical-adapter"
    gpu_memory_utilization: float = 0.88
    max_model_len: Optional[int] = None
    trust_remote_code: bool = False

    def validate(self):
        if not self.base_model:
            raise ValueError("base_model cannot be empty")
        if len(self.gpu_ids) < 1:
            raise ValueError("At least one GPU must be selected")
        if self.strategy not in {"data_parallel", "tensor_parallel"}:
            raise ValueError("strategy must be data_parallel or tensor_parallel")
        if self.precision not in {"bf16", "int4"}:
            raise ValueError("precision must be bf16 or int4")
        if not (0.1 <= self.gpu_memory_utilization <= 0.98):
            raise ValueError("gpu_memory_utilization must be between 0.1 and 0.98")
        if not (1 <= int(self.port) <= 65535):
            raise ValueError("port must be in 1..65535")
        return self

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def to_json_dict(self):
        payload = asdict(self)
        payload["gpu_ids"] = list(self.gpu_ids)
        if self.adapter_path is not None:
            payload["adapter_path"] = str(self.adapter_path)
        return payload


@dataclass(frozen=True)
class RAGConfig:
    enabled: bool = False
    backend: str = "local_hybrid"
    embedding_backend: str = "local"  # local | hf | hash; ModelScope auto-download disabled during KB build
    embedding_model: Optional[str] = field(default_factory=default_local_embedding_model)
    top_k: int = 5
    graph_hops: int = 1
    chunk_size: int = 1200
    chunk_overlap: int = 150
    context_max_chars: int = 9000
    knowledge_source_dir: Optional[Path] = None
    knowledge_base_dir: Optional[Path] = None

    def validate(self):
        if self.embedding_backend not in {"hash", "hf", "modelscope", "local"}:
            raise ValueError("embedding_backend must be modelscope, local, hf, or hash")
        if self.enabled and self.embedding_backend != "hash" and not self.embedding_model:
            raise ValueError("embedding_model is required for semantic embeddings")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.graph_hops not in {0, 1, 2}:
            raise ValueError("graph_hops must be 0, 1, or 2")
        if self.chunk_size < 256:
            raise ValueError("chunk_size must be >= 256")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >=0 and < chunk_size")
        if self.context_max_chars < 1000:
            raise ValueError("context_max_chars must be >= 1000")
        return self
