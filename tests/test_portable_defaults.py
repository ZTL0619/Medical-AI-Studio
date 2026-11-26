"""Portable configuration regression checks for GitHub distribution."""

from pathlib import Path

from foundation_model.config import RAGConfig
from foundation_model.rag.defaults import default_local_embedding_model


def test_rag_model_path_requires_explicit_input(monkeypatch):
    monkeypatch.delenv("MEDICAL_RAG_EMBEDDING_MODEL", raising=False)
    assert default_local_embedding_model() == ""
    assert RAGConfig(enabled=False).validate().enabled is False
    try:
        RAGConfig(enabled=True).validate()
    except ValueError as exc:
        assert "embedding_model" in str(exc)
    else:
        raise AssertionError("A new RAG index needs an explicit embedding model")


def test_rag_model_path_can_be_provided_by_environment(monkeypatch, tmp_path):
    path = tmp_path / "my_local_bge"
    monkeypatch.setenv("MEDICAL_RAG_EMBEDDING_MODEL", str(path))
    assert RAGConfig(enabled=True).validate().embedding_model == str(path)
    assert default_local_embedding_model() == str(path)


def test_no_lilab_data_in_runtime_sources():
    runtime_files = (
        Path("foundation_model/config.py"),
        Path("foundation_model/models/registry.py"),
        Path("studio/foundation_config_store.py"),
        Path("studio/ui/foundation_sidebar.py"),
    )
    for path in runtime_files:
        assert "/lilab_data/" not in path.read_text(encoding="utf-8"), path


def test_lingshu_requires_user_model_path():
    from foundation_model.models.registry import get_foundation_model
    assert get_foundation_model("lingshu7b").default_model_id == ""
