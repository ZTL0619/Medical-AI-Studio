"""Embedding backends used by the local Hybrid GraphRAG index."""
from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _emit(callback: Callable[[dict], None] | None, **event) -> None:
    if callback:
        callback(event)


def validate_local_model_dir(model_dir: str | Path | None) -> dict:
    """Validate a local Transformers embedding model without touching the network."""
    if not model_dir:
        return {"valid": False, "path": "", "error": "尚未选择本地 Embedding 模型目录。"}
    path = Path(model_dir).expanduser().resolve()
    if not path.is_dir():
        return {"valid": False, "path": str(path), "error": "模型目录不存在。"}

    config_ok = (path / "config.json").is_file()
    weight_candidates = [path / "model.safetensors", path / "pytorch_model.bin"]
    weight_ok = any(p.is_file() for p in weight_candidates) or any(path.glob("model-*.safetensors")) or any(path.glob("pytorch_model-*.bin"))
    tokenizer_candidates = [
        path / "tokenizer.json",
        path / "tokenizer_config.json",
        path / "vocab.txt",
        path / "vocab.json",
        path / "sentencepiece.bpe.model",
        path / "spiece.model",
    ]
    tokenizer_ok = any(p.is_file() for p in tokenizer_candidates)

    missing = []
    if not config_ok:
        missing.append("config.json")
    if not weight_ok:
        missing.append("model.safetensors / pytorch_model.bin")
    if not tokenizer_ok:
        missing.append("tokenizer files")
    if missing:
        return {
            "valid": False,
            "path": str(path),
            "error": "模型目录不完整，缺少：" + ", ".join(missing),
        }
    return {"valid": True, "path": str(path)}


class HashingTextEncoder:
    """Deterministic multilingual char/word hashing encoder."""

    def __init__(self, dimension: int = 768):
        self.dimension = int(dimension)
        self.backend = "hash"
        self.model_name = None

    def _features(self, text: str) -> Iterable[str]:
        text = str(text or "").lower()
        for token in _WORD_RE.findall(text):
            yield "w:" + token
        for run in _CJK_RE.findall(text):
            run = re.sub(r"\s+", "", run)
            for n in (2, 3, 4):
                if len(run) < n:
                    continue
                for i in range(len(run) - n + 1):
                    yield f"c{n}:" + run[i:i+n]

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int = 32) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little", signed=False)
                index = value % self.dimension
                sign = 1.0 if ((value >> 9) & 1) else -1.0
                counts[index] = counts.get(index, 0.0) + sign
            for index, value in counts.items():
                matrix[row, index] = math.copysign(math.log1p(abs(value)), value)
        return _l2_normalize(matrix)


class HFTextEncoder:
    """Transformers encoder loaded from a model id or a local directory."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        *,
        source_name: str | None = None,
        local_files_only: bool = False,
        progress_callback: Callable[[dict], None] | None = None,
    ):
        _emit(progress_callback, phase="embedding_model", progress=0.465, message="正在导入 Embedding 运行库……")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.backend = "hf"
        self.model_name = str(source_name or model_name)
        self.model_path = str(model_name)
        self.device = str(device)

        _emit(progress_callback, phase="embedding_tokenizer", progress=0.475, message="正在加载 Embedding tokenizer……")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=local_files_only)

        _emit(progress_callback, phase="embedding_weights", progress=0.490, message="正在从本地磁盘加载 Embedding 模型权重……")
        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=local_files_only,
            use_safetensors=False,
        ).to(self.device)
        self.model.eval()
        self.dimension = int(getattr(self.model.config, "hidden_size", 0) or 0)
        _emit(progress_callback, phase="embedding_ready", progress=0.510, message="Embedding 模型已就绪，开始生成向量……")

    def _prepare(self, texts: list[str], is_query: bool) -> list[str]:
        name = self.model_name.lower()
        if "e5" in name:
            prefix = "query: " if is_query else "passage: "
            return [prefix + str(text) for text in texts]
        if "bge" in name and is_query:
            is_chinese_bge = any(tag in name for tag in ("-zh", "_zh", "/zh", "chinese"))
            prefix = (
                "为这个句子生成表示以用于检索相关文章："
                if is_chinese_bge
                else "Represent this sentence for searching relevant passages: "
            )
            return [prefix + str(text) for text in texts]
        return [str(text) for text in texts]

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int = 16) -> np.ndarray:
        rows: list[np.ndarray] = []
        prepared = self._prepare(texts, is_query)
        use_cls = "bge" in self.model_name.lower()
        with self.torch.inference_mode():
            for start in range(0, len(prepared), int(batch_size)):
                batch = prepared[start:start + int(batch_size)]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                output = self.model(**encoded)
                hidden = output.last_hidden_state
                if use_cls:
                    pooled = hidden[:, 0]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                rows.append(pooled.detach().cpu().float().numpy())
        if not rows:
            return np.zeros((0, self.dimension or 1), dtype=np.float32)
        return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


class ModelScopeTextEncoder(HFTextEncoder):
    """Deprecated online ModelScope loader.

    Knowledge-base construction deliberately does not download models anymore.
    Download/upload the model first and use the local backend. This avoids a
    network call being hidden behind the 'loading embedding model' stage.
    """

    def __init__(self, model_name: str, device: str = "cpu", **kwargs):
        raise RuntimeError(
            "建库阶段已禁用自动 ModelScope 下载，避免长时间卡在“加载 Embedding 模型”。"
            "请先把 BAAI/bge-small-en-v1.5 下载/上传到服务器，再选择“服务器本地模型”。"
        )


class LocalTextEncoder(HFTextEncoder):
    def __init__(self, model_name: str, device: str = "cpu", *, progress_callback: Callable[[dict], None] | None = None):
        validation = validate_local_model_dir(model_name)
        _emit(progress_callback, phase="embedding_check", progress=0.460, message="正在检查本地 Embedding 模型文件……")
        if not validation.get("valid"):
            raise FileNotFoundError(f"本地 Embedding 模型不可用：{validation.get('error')} 路径：{validation.get('path')}")
        path = Path(validation["path"])
        # ModelScope snapshots often end in snapshots/master, which is not the
        # model identity. Recover the repository name from cache parents so BGE
        # keeps CLS pooling and the English BGE query instruction after loading
        # from a local snapshot path.
        source_name = path.name
        for candidate in (path, *path.parents):
            candidate_name = candidate.name
            if "--" in candidate_name and candidate_name.lower().endswith("bge-small-en-v1.5"):
                owner, repo = candidate_name.split("--", 1)
                source_name = f"{owner}/{repo}"
                break
        if source_name in {"master", "main", "snapshots"}:
            source_name = "BAAI/bge-small-en-v1.5"
        super().__init__(
            str(path),
            device=device,
            source_name=source_name,
            local_files_only=True,
            progress_callback=progress_callback,
        )
        self.backend = "local"


def create_encoder(
    backend: str,
    model_name: str | None = None,
    *,
    dimension: int = 768,
    device: str = "cpu",
    progress_callback: Callable[[dict], None] | None = None,
):
    backend = str(backend or "hash").lower()
    if backend == "hash":
        return HashingTextEncoder(dimension=dimension)
    if backend == "hf":
        if not model_name:
            raise ValueError("Hugging Face 向量模式需要 embedding model。")
        return HFTextEncoder(model_name=model_name, device=device, progress_callback=progress_callback)
    if backend == "modelscope":
        if not model_name:
            raise ValueError("ModelScope 向量模式需要 embedding model。")
        return ModelScopeTextEncoder(model_name=model_name, device=device, progress_callback=progress_callback)
    if backend == "local":
        if not model_name:
            raise ValueError("本地向量模式需要模型目录。")
        return LocalTextEncoder(model_name=model_name, device=device, progress_callback=progress_callback)
    raise ValueError(f"Unsupported embedding backend: {backend}")
