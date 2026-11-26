import json
from pathlib import Path

from foundation_model.rag.index import GraphRAGIndexManager


def test_builder_outputs_portable_snippets_report_and_valid_index(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.txt").write_text(
        ("Pulmonary consolidation can be associated with pneumonia. "
         "Chest radiography findings should be interpreted with clinical context.\n") * 20,
        encoding="utf-8",
    )
    index = tmp_path / "kb"
    events = []
    manager = GraphRAGIndexManager(index)
    report = manager.build(
        docs,
        chunk_size=320,
        chunk_overlap=40,
        embedding_backend="hash",
        progress_callback=events.append,
    )
    assert report.documents == 1
    assert report.chunks > 1
    assert (index / "snippets.jsonl").exists()
    assert (index / "build_report.json").exists()
    assert manager.validate()["valid"] is True
    assert events and events[-1]["phase"] == "done"
    rows = [json.loads(x) for x in (index / "snippets.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert rows[0]["content"] == rows[0]["text"]


def test_image_only_pdf_is_reported_and_skipped(tmp_path):
    from pypdf import PdfWriter

    docs = tmp_path / "docs"
    docs.mkdir()
    writer = PdfWriter()
    for _ in range(4):
        writer.add_blank_page(width=300, height=300)
    with (docs / "scan.pdf").open("wb") as f:
        writer.write(f)
    # Keep one real text document so the complete build can proceed.
    (docs / "usable.txt").write_text("B cell lymphoma pathology CD20 marker. " * 50, encoding="utf-8")

    manager = GraphRAGIndexManager(tmp_path / "kb")
    report = manager.build(docs, chunk_size=300, chunk_overlap=20, embedding_backend="hash")
    assert report.documents == 1
    assert report.scanned_pdf_files == 1
    assert any("scan.pdf" in x for x in report.scanned_files)
    assert report.pdf_pages_skipped >= 4


def test_sidebar_starts_with_build_or_import_flow():
    sidebar = Path("studio/ui/foundation_sidebar.py").read_text(encoding="utf-8")
    assert "构建医学语料库" in sidebar
    assert "导入已有语料库" in sidebar
    assert "先建立或导入一个医学语料库" in sidebar
    assert "医学语料库已加载" in sidebar
