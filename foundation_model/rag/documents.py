"""Document ingestion and conservative text cleaning for medical GraphRAG.

The builder deliberately does **not** OCR image-only/scanned PDFs.  Such PDFs
are reported and skipped so low-quality OCR text cannot silently contaminate a
medical knowledge base.  Mixed PDFs remain usable: pages with a real text layer
are kept while image-only pages are skipped and counted.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Callable


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".docx", ".json", ".jsonl"}
MIN_PDF_PAGE_TEXT_CHARS = 24


@dataclass(frozen=True)
class SourceSegment:
    text: str
    source: str
    metadata: dict


@dataclass
class IngestionReport:
    input_files: int = 0
    accepted_files: int = 0
    scanned_pdf_files: int = 0
    failed_files: int = 0
    empty_files: int = 0
    unsupported_files: int = 0
    pdf_pages_total: int = 0
    pdf_pages_with_text: int = 0
    pdf_pages_skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    scanned_files: list[str] = field(default_factory=list)
    failed_file_details: list[dict] = field(default_factory=list)
    accepted_file_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_text(text: str) -> str:
    """Conservative cleaning that avoids rewriting medical meaning."""
    text = str(text or "").replace("\x00", " ").replace("\ufeff", "")
    # Repair English words split only by PDF line wrapping: diagnos-\ntic.
    text = re.sub(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_page_number(line: str) -> bool:
    line = line.strip()
    return bool(
        re.fullmatch(r"(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?", line, flags=re.I)
        or re.fullmatch(r"[-–—]?\s*\d+\s*[-–—]?", line)
    )


def _nonempty_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line).strip() for line in str(text or "").splitlines() if line.strip()]


def _repeated_pdf_margin_lines(raw_pages: list[str]) -> set[str]:
    """Find only strongly repeated short header/footer lines.

    A line must occur in the first/last two non-empty lines on at least 60% of
    pages (minimum 3 pages).  This deliberately errs on the side of keeping text.
    """
    if len(raw_pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for raw in raw_pages:
        lines = _nonempty_lines(raw)
        candidates = set(lines[:2] + lines[-2:])
        for line in candidates:
            if 2 <= len(line) <= 180 and not _is_page_number(line):
                counts[line] += 1
    threshold = max(3, int(math.ceil(len(raw_pages) * 0.60)))
    return {line for line, count in counts.items() if count >= threshold}


def _clean_pdf_page(raw: str, repeated_margin_lines: set[str]) -> str:
    lines = _nonempty_lines(raw)
    cleaned: list[str] = []
    for line in lines:
        if line in repeated_margin_lines or _is_page_number(line):
            continue
        cleaned.append(line)
    return _clean_text("\n".join(cleaned))


def _flatten_json(value, prefix="") -> list[str]:
    parts: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            parts.extend(_flatten_json(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            parts.extend(_flatten_json(item, f"{prefix}[{index}]"))
    elif value is not None:
        text = _clean_text(value)
        if text:
            parts.append(f"{prefix}: {text}" if prefix else text)
    return parts


def _read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("word/document.xml")
    root = ET.fromstring(raw)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", ns)]
        joined = _clean_text("".join(texts))
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def _source_name(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_pdf(path: Path, source: str, report: IngestionReport) -> list[SourceSegment]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("读取 PDF 需要 pypdf；请执行 `uv pip install pypdf`。") from exc

    reader = PdfReader(str(path))
    raw_pages: list[str] = []
    for page in reader.pages:
        try:
            raw_pages.append(page.extract_text() or "")
        except Exception:
            raw_pages.append("")

    page_count = len(raw_pages)
    report.pdf_pages_total += page_count
    repeated = _repeated_pdf_margin_lines(raw_pages)
    cleaned_pages = [_clean_pdf_page(raw, repeated) for raw in raw_pages]
    effective_pages = [text for text in cleaned_pages if len(re.sub(r"\s+", "", text)) >= MIN_PDF_PAGE_TEXT_CHARS]
    text_page_count = len(effective_pages)
    total_chars = sum(len(re.sub(r"\s+", "", text)) for text in effective_pages)

    # Image-only/scanned detection. Mixed PDFs are *not* rejected: if enough
    # pages have a usable text layer, those pages are retained.
    if page_count <= 1:
        scanned = page_count == 1 and total_chars < 80
    else:
        ratio = text_page_count / max(1, page_count)
        scanned = ratio < 0.15 and total_chars < max(240, page_count * 24)

    if scanned:
        report.scanned_pdf_files += 1
        report.scanned_files.append(source)
        report.pdf_pages_skipped += page_count
        report.warnings.append(f"扫描/图像型 PDF 已跳过：{source}（未检测到足够的文本层）")
        return []

    segments: list[SourceSegment] = []
    for page_index, text in enumerate(cleaned_pages, 1):
        compact_len = len(re.sub(r"\s+", "", text))
        if compact_len < MIN_PDF_PAGE_TEXT_CHARS:
            report.pdf_pages_skipped += 1
            continue
        report.pdf_pages_with_text += 1
        segments.append(SourceSegment(text=text, source=source, metadata={"page": page_index}))
    return segments


def read_document(
    path: str | Path,
    *,
    root: str | Path | None = None,
    report: IngestionReport | None = None,
) -> tuple[list[SourceSegment], list[str]]:
    """Read one supported document and preserve provenance metadata."""
    path = Path(path).expanduser().resolve()
    root_path = Path(root).expanduser().resolve() if root else path.parent
    source = _source_name(path, root_path)
    report = report or IngestionReport()

    suffix = path.suffix.lower()
    segments: list[SourceSegment] = []
    warnings: list[str] = []

    try:
        if suffix == ".pdf":
            segments = _read_pdf(path, source, report)

        elif suffix == ".docx":
            text = _clean_text(_read_docx(path))
            if text:
                segments.append(SourceSegment(text=text, source=source, metadata={}))

        elif suffix in {".txt", ".md", ".markdown"}:
            text = _clean_text(path.read_text(encoding="utf-8", errors="ignore"))
            if text:
                segments.append(SourceSegment(text=text, source=source, metadata={}))

        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            text = _clean_text("\n".join(_flatten_json(payload)))
            if text:
                segments.append(SourceSegment(text=text, source=source, metadata={}))

        elif suffix == ".jsonl":
            for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    text = _clean_text("\n".join(_flatten_json(payload)))
                except json.JSONDecodeError:
                    text = _clean_text(line)
                if text:
                    segments.append(SourceSegment(text=text, source=source, metadata={"line": line_no}))
        else:
            warnings.append(f"跳过不支持的文件：{source}")
    except Exception as exc:
        warnings.append(f"读取失败 {source}: {exc}")

    return segments, warnings


def collect_documents_with_report(
    source_dir: str | Path,
    *,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[list[SourceSegment], list[str], IngestionReport]:
    """Collect supported documents plus detailed ingestion quality statistics."""
    root = Path(source_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"医学语料目录不存在：{root}")

    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    report = IngestionReport(input_files=len(paths))
    segments: list[SourceSegment] = []
    accepted: list[str] = []

    if progress_callback:
        progress_callback({"phase": "scan", "current": 0, "total": len(paths), "message": f"发现 {len(paths)} 个可处理文件"})

    for i, path in enumerate(paths, 1):
        source = _source_name(path, root)
        before_scanned = report.scanned_pdf_files
        docs, issues = read_document(path, root=root, report=report)
        report.warnings.extend(issues)
        if issues:
            report.failed_files += 1
            report.failed_file_details.append({"source": source, "reason": issues[0]})
        elif docs:
            report.accepted_files += 1
            report.accepted_file_paths.append(str(path))
            accepted.append(str(path))
            segments.extend(docs)
        elif report.scanned_pdf_files > before_scanned:
            pass  # already counted as scanned, not as a generic failure
        else:
            report.empty_files += 1
            report.warnings.append(f"无可用文本，已跳过：{source}")
        if progress_callback:
            progress_callback({"phase": "parse", "current": i, "total": len(paths), "message": f"正在解析：{source}"})

    return segments, accepted, report


def collect_documents(source_dir: str | Path) -> tuple[list[SourceSegment], list[str], list[str]]:
    """Backward-compatible collector used by older callers/tests."""
    segments, files, report = collect_documents_with_report(source_dir)
    return segments, files, list(report.warnings)
