"""Build, validate and manage a local medical Hybrid GraphRAG knowledge base."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Callable, Iterable

import numpy as np

from foundation_model.rag.documents import SourceSegment, collect_documents_with_report
from foundation_model.rag.embeddings import create_encoder


_EN_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "were", "was", "are", "into", "have", "has",
    "had", "not", "but", "can", "may", "also", "than", "then", "when", "where", "which", "who", "what",
    "how", "why", "its", "their", "these", "those", "using", "used", "use", "patient", "patients", "image",
    "images", "figure", "fig", "table", "study", "result", "results", "data", "based", "show", "shows",
}
_CN_STOP = {
    "这个", "这些", "那个", "以及", "或者", "因为", "所以", "如果", "可以", "可能", "进行", "相关", "其中",
    "一个", "一种", "通过", "对于", "患者", "图像", "显示", "结果", "研究", "使用", "包括", "主要", "需要",
    "由于", "没有", "具有", "出现", "存在", "能够", "分析", "临床", "医学", "检查", "治疗", "诊断",
}
_EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-/]{2,}")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}")


@dataclass(frozen=True)
class GraphRAGBuildReport:
    source_dir: str
    index_dir: str
    input_files: int
    documents: int
    scanned_pdf_files: int
    failed_files: int
    empty_files: int
    pdf_pages_total: int
    pdf_pages_with_text: int
    pdf_pages_skipped: int
    segments: int
    chunks: int
    entities: int
    relations: int
    embedding_backend: str
    embedding_model: str | None
    warnings: list[str]
    scanned_files: list[str]
    failed_file_details: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", str(term or "").strip().lower())


def extract_concepts(text: str, max_terms: int = 24) -> list[str]:
    """Extract compact multilingual concept nodes for the lightweight graph."""
    counter: Counter[str] = Counter()
    raw = str(text or "")
    for token in _EN_RE.findall(raw):
        term = _normalize_term(token)
        if term not in _EN_STOP and not term.isdigit():
            counter[term] += 1

    cjk_runs = _CJK_RUN_RE.findall(raw)
    try:
        import jieba  # type: ignore
        for run in cjk_runs:
            for token in jieba.cut(run, cut_all=False):
                term = _normalize_term(token)
                if 2 <= len(term) <= 16 and term not in _CN_STOP:
                    counter[term] += 1
    except ImportError:
        for run in cjk_runs:
            if 2 <= len(run) <= 12:
                counter[run] += 1
            else:
                for n in (2, 3, 4):
                    for i in range(max(0, len(run) - n + 1)):
                        term = run[i:i+n]
                        if term not in _CN_STOP:
                            counter[term] += 0.35

    def score(item):
        term, count = item
        return float(count) * (1.0 + min(len(term), 10) / 20.0)

    ranked = sorted(counter.items(), key=lambda item: (-score(item), item[0]))
    return [term for term, _ in ranked[: int(max_terms)]]


def _chunk_segment(segment: SourceSegment, chunk_size: int, overlap: int) -> Iterable[tuple[str, dict]]:
    text = segment.text.strip()
    if not text:
        return
    chunk_size = max(256, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2))
    start = 0
    chunk_index = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            floor = start + int(chunk_size * 0.65)
            candidates = [text.rfind(mark, floor, end) for mark in ("\n\n", "。", ". ", "; ", "；")]
            split = max(candidates)
            if split > start:
                end = split + 1
        piece = text[start:end].strip()
        if piece:
            metadata = dict(segment.metadata)
            metadata["chunk_index"] = chunk_index
            metadata["char_start"] = start
            metadata["char_end"] = end
            yield piece, metadata
            chunk_index += 1
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def _chunk_id(source: str, metadata: dict, text: str) -> str:
    raw = json.dumps([source, metadata, text[:160]], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _progress(callback: Callable[[dict], None] | None, *, phase: str, progress: float, message: str, current=None, total=None):
    if callback:
        callback({
            "phase": phase,
            "progress": max(0.0, min(1.0, float(progress))),
            "message": str(message),
            "current": current,
            "total": total,
        })


class GraphRAGIndexManager:
    MANIFEST = "manifest.json"
    CHUNKS = "chunks.jsonl"          # runtime/backward-compatible name
    SNIPPETS = "snippets.jsonl"      # portable user-facing corpus artifact
    GRAPH = "graph.json"
    EMBEDDINGS = "embeddings.npy"
    BUILD_REPORT = "build_report.json"

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir).expanduser().resolve()

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / self.MANIFEST

    @property
    def ready(self) -> bool:
        return all((self.index_dir / name).exists() for name in (self.MANIFEST, self.CHUNKS, self.GRAPH, self.EMBEDDINGS))

    def status(self) -> dict:
        if not self.ready:
            return {"ready": False, "index_dir": str(self.index_dir)}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            payload["ready"] = True
            payload["index_dir"] = str(self.index_dir)
            payload["fingerprint"] = self.fingerprint()
            payload["portable_snippets"] = (self.index_dir / self.SNIPPETS).exists()
            payload["build_report_available"] = (self.index_dir / self.BUILD_REPORT).exists()
            return payload
        except Exception as exc:
            return {"ready": False, "index_dir": str(self.index_dir), "error": str(exc)}

    def validate(self) -> dict:
        """Validate an existing knowledge-base directory before importing it."""
        if not self.ready:
            missing = [name for name in (self.MANIFEST, self.CHUNKS, self.GRAPH, self.EMBEDDINGS) if not (self.index_dir / name).exists()]
            return {"valid": False, "index_dir": str(self.index_dir), "error": "缺少必要文件：" + ", ".join(missing)}
        try:
            manifest = json.loads((self.index_dir / self.MANIFEST).read_text(encoding="utf-8"))
            with (self.index_dir / self.CHUNKS).open("r", encoding="utf-8") as handle:
                chunk_count = sum(1 for line in handle if line.strip())
            embeddings = np.load(self.index_dir / self.EMBEDDINGS, mmap_mode="r")
            graph = json.loads((self.index_dir / self.GRAPH).read_text(encoding="utf-8"))
            if embeddings.ndim != 2:
                raise ValueError("embeddings.npy 不是二维向量矩阵")
            if int(embeddings.shape[0]) != int(chunk_count):
                raise ValueError(f"snippet 数量 ({chunk_count}) 与 embedding 数量 ({embeddings.shape[0]}) 不一致")
            if not isinstance(graph.get("entity_to_chunks", {}), dict):
                raise ValueError("graph.json 缺少 entity_to_chunks")
            return {
                "valid": True,
                "index_dir": str(self.index_dir),
                "documents": int(manifest.get("documents", 0)),
                "chunks": int(manifest.get("chunks", chunk_count)),
                "entities": int(manifest.get("entities", 0)),
                "relations": int(manifest.get("relations", 0)),
                "embedding_backend": manifest.get("embedding_backend", "unknown"),
                "embedding_model": manifest.get("embedding_model"),
                "version": int(manifest.get("version", 1)),
            }
        except Exception as exc:
            return {"valid": False, "index_dir": str(self.index_dir), "error": str(exc)}

    def build_report(self) -> dict | None:
        path = self.index_dir / self.BUILD_REPORT
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def fingerprint(self) -> str:
        if not self.manifest_path.exists():
            return "missing"
        stat = self.manifest_path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def clear(self):
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        return {"ready": False, "index_dir": str(self.index_dir)}

    def build(
        self,
        source_dir: str | Path,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        embedding_backend: str = "hash",
        embedding_model: str | None = None,
        embedding_device: str = "cpu",
        batch_size: int = 16,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> GraphRAGBuildReport:
        source_dir = Path(source_dir).expanduser().resolve()
        _progress(progress_callback, phase="scan", progress=0.01, message="正在扫描医学语料目录……")

        def ingest_progress(event: dict):
            cur = int(event.get("current") or 0)
            total = max(1, int(event.get("total") or 1))
            frac = 0.03 + 0.22 * (cur / total)
            _progress(
                progress_callback,
                phase=str(event.get("phase") or "parse"),
                progress=frac,
                message=str(event.get("message") or "正在解析医学文档……"),
                current=cur,
                total=total,
            )

        segments, files, ingestion = collect_documents_with_report(source_dir, progress_callback=ingest_progress)
        if not files:
            if ingestion.scanned_pdf_files and not ingestion.failed_files:
                raise RuntimeError("没有可用文本语料：检测到的 PDF 均为扫描/图像型 PDF。当前版本不做 OCR，请换用带文本层的 PDF。")
            raise RuntimeError("语料目录中没有可索引的 PDF/TXT/MD/DOCX/JSON/JSONL 文本。")

        _progress(progress_callback, phase="chunk", progress=0.26, message="正在清洗并生成 snippets……")
        chunks: list[dict] = []
        entity_counts: Counter[str] = Counter()
        entity_to_chunks: dict[str, list[int]] = defaultdict(list)
        edge_weights: dict[str, Counter[str]] = defaultdict(Counter)

        total_segments = max(1, len(segments))
        for seg_index, segment in enumerate(segments, 1):
            for text, metadata in _chunk_segment(segment, chunk_size, chunk_overlap):
                concepts = extract_concepts(text)
                chunk_idx = len(chunks)
                chunk = {
                    "chunk_id": _chunk_id(segment.source, metadata, text),
                    "text": text,
                    "content": text,  # portable corpus naming
                    "source": segment.source,
                    "metadata": metadata,
                    "entities": concepts,
                }
                chunks.append(chunk)
                unique = list(dict.fromkeys(concepts))
                for entity in unique:
                    entity_counts[entity] += 1
                    entity_to_chunks[entity].append(chunk_idx)
                for i, left in enumerate(unique):
                    for right in unique[i + 1:]:
                        edge_weights[left][right] += 1
                        edge_weights[right][left] += 1
            if seg_index == total_segments or seg_index % max(1, total_segments // 25) == 0:
                _progress(
                    progress_callback,
                    phase="chunk",
                    progress=0.26 + 0.19 * (seg_index / total_segments),
                    message=f"正在生成 snippets：{seg_index}/{total_segments} 个文本片段",
                    current=seg_index,
                    total=total_segments,
                )

        if not chunks:
            raise RuntimeError("文档读取成功，但没有提取到可索引文本。")

        _progress(progress_callback, phase="embedding_check", progress=0.46, message="正在检查 Embedding 模型……")
        encoder = create_encoder(
            embedding_backend,
            embedding_model,
            device=embedding_device,
            progress_callback=progress_callback,
        )
        vectors: list[np.ndarray] = []
        total_chunks = len(chunks)
        for start in range(0, total_chunks, max(1, int(batch_size))):
            end = min(total_chunks, start + max(1, int(batch_size)))
            batch = [item["text"] for item in chunks[start:end]]
            vectors.append(encoder.encode(batch, is_query=False, batch_size=max(1, int(batch_size))))
            _progress(
                progress_callback,
                phase="embedding",
                progress=0.52 + 0.20 * (end / total_chunks),
                message=f"正在生成 Embedding：{end}/{total_chunks} snippets",
                current=end,
                total=total_chunks,
            )
        embeddings = np.concatenate(vectors, axis=0)
        if embeddings.shape[0] != len(chunks):
            raise RuntimeError("向量编码结果与文本切块数量不一致。")

        _progress(progress_callback, phase="graph", progress=0.74, message="正在整理医学概念图关系……")
        graph_payload = {
            "entity_counts": dict(entity_counts),
            "entity_to_chunks": dict(entity_to_chunks),
            "neighbors": {
                entity: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:96]
                for entity, counter in edge_weights.items()
            },
        }
        relations = sum(len(v) for v in edge_weights.values()) // 2

        self.index_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_dir / ".building"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)

        _progress(progress_callback, phase="save", progress=0.86, message="正在保存知识库文件……")
        for filename in (self.CHUNKS, self.SNIPPETS):
            with (tmp / filename).open("w", encoding="utf-8") as handle:
                for chunk in chunks:
                    handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        (tmp / self.GRAPH).write_text(json.dumps(graph_payload, ensure_ascii=False), encoding="utf-8")
        np.save(tmp / self.EMBEDDINGS, embeddings.astype(np.float32, copy=False))

        report = GraphRAGBuildReport(
            source_dir=str(source_dir),
            index_dir=str(self.index_dir),
            input_files=int(ingestion.input_files),
            documents=len(files),
            scanned_pdf_files=int(ingestion.scanned_pdf_files),
            failed_files=int(ingestion.failed_files),
            empty_files=int(ingestion.empty_files),
            pdf_pages_total=int(ingestion.pdf_pages_total),
            pdf_pages_with_text=int(ingestion.pdf_pages_with_text),
            pdf_pages_skipped=int(ingestion.pdf_pages_skipped),
            segments=len(segments),
            chunks=len(chunks),
            entities=len(entity_counts),
            relations=int(relations),
            embedding_backend=str(embedding_backend),
            embedding_model=str(embedding_model) if embedding_model else None,
            warnings=list(ingestion.warnings),
            scanned_files=list(ingestion.scanned_files),
            failed_file_details=list(ingestion.failed_file_details),
        )
        manifest = {
            "version": 2,
            "format": "decodeplex-medical-graphrag",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(source_dir),
            "documents": report.documents,
            "input_files": report.input_files,
            "segments": report.segments,
            "chunks": report.chunks,
            "entities": report.entities,
            "relations": report.relations,
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "embedding_backend": str(embedding_backend),
            "embedding_model": str(embedding_model) if embedding_model else None,
            "embedding_resolved_path": str(getattr(encoder, "local_path", "") or getattr(encoder, "model_path", "")) or None,
            "embedding_dimension": int(embeddings.shape[1]),
            "scanned_pdf_files": report.scanned_pdf_files,
            "failed_files": report.failed_files,
            "pdf_pages_total": report.pdf_pages_total,
            "pdf_pages_with_text": report.pdf_pages_with_text,
            "pdf_pages_skipped": report.pdf_pages_skipped,
            "warnings": report.warnings,
        }
        (tmp / self.MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (tmp / self.BUILD_REPORT).write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        for name in (self.CHUNKS, self.SNIPPETS, self.GRAPH, self.EMBEDDINGS, self.MANIFEST, self.BUILD_REPORT):
            src = tmp / name
            dst = self.index_dir / name
            src.replace(dst)
        shutil.rmtree(tmp, ignore_errors=True)
        _progress(progress_callback, phase="done", progress=1.0, message="医学知识库构建完成。")
        return report
