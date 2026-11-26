from pathlib import Path


def test_local_english_bge_is_default_and_stop_button_exists():
    sidebar = Path("studio/ui/foundation_sidebar.py").read_text(encoding="utf-8")
    store = Path("studio/foundation_config_store.py").read_text(encoding="utf-8")
    config = Path("foundation_model/config.py").read_text(encoding="utf-8")
    embeddings = Path("foundation_model/rag/embeddings.py").read_text(encoding="utf-8")
    worker = Path("medical_kb_worker.py").read_text(encoding="utf-8")
    jobs = Path("foundation_model/rag/build_jobs.py").read_text(encoding="utf-8")

    assert "BAAI/bge-small-en-v1.5" in sidebar
    assert "default_local_embedding_model" in store
    assert "default_local_embedding_model" in config
    assert '"rag_embedding_backend": "local"' in store
    assert "停止构建" in sidebar
    assert "MedicalKBJobManager" in sidebar
    assert "validate_local_model_dir" in embeddings
    assert "建库阶段已禁用自动 ModelScope 下载" in embeddings
    assert "Represent this sentence for searching relevant passages:" in embeddings
    assert "GraphRAGIndexManager" in worker
    assert "subprocess.Popen" in jobs
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in worker
    assert "backend != \"local\"" in worker
    assert "backend != \"local\"" in jobs
