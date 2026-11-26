from pathlib import Path

from foundation_model.rag.index import GraphRAGIndexManager
from foundation_model.rag.local_backend import LocalHybridMedicalKnowledgeBase
from foundation_model.rag.pipeline import HybridGraphRAGPipeline


def test_local_hybrid_graphrag_build_retrieve_and_citations(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "lymphoma.txt").write_text(
        "B cell lymphoma is a lymphoid malignancy. CD20 positive large B cells may be seen in diffuse large B-cell lymphoma. "
        "Histopathology can show diffuse sheets of atypical large lymphoid cells.\n" * 5,
        encoding="utf-8",
    )
    (docs / "cardiology.txt").write_text(
        "Aortic stenosis is evaluated by echocardiography and transvalvular gradients. Valve calcification may be present.\n" * 5,
        encoding="utf-8",
    )

    manager = GraphRAGIndexManager(tmp_path / "index")
    report = manager.build(docs, chunk_size=300, chunk_overlap=40, embedding_backend="hash")
    assert report.documents == 2
    assert report.chunks >= 2
    assert report.entities > 0
    assert report.relations > 0
    assert manager.ready is True

    kb = LocalHybridMedicalKnowledgeBase(tmp_path / "index")
    pipeline = HybridGraphRAGPipeline(kb)
    context = pipeline.retrieve("CD20 B cell lymphoma histopathology", top_k=2, graph_hops=1)
    assert context.sources
    assert context.sources[0]["source"] == "lymphoma.txt"
    assert context.sources[0]["retrieval_path"] in {"vector", "hybrid"}
    prompt = pipeline.augment_prompt("What is the diagnosis?", context)
    assert "医学知识证据" in prompt
    assert "[1]" in prompt


def test_foundation_ui_wires_graphrag_into_inference():
    sidebar = Path("studio/ui/foundation_sidebar.py").read_text(encoding="utf-8")
    page = Path("studio/foundation_page.py").read_text(encoding="utf-8")
    assert "构建医学语料库" in sidebar
    assert "导入已有语料库" in sidebar
    assert "检索测试" in sidebar
    assert "GraphRAGIndexManager" in sidebar
    assert "_retrieve_multimodal_context" in page
    assert "build_visual_clue_prompt" in page
    assert "build_multimodal_query" in page
    assert "build_final_prompt" in page
    assert "GraphRAG 检索过程与医学证据" in page
