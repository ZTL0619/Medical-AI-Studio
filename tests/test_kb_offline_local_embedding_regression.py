"""The KB builder must be portable and remain offline-only."""
from pathlib import Path


def test_builder_has_no_machine_specific_embedding_default():
    store = Path("studio/foundation_config_store.py").read_text(encoding="utf-8")
    config = Path("foundation_model/config.py").read_text(encoding="utf-8")
    sidebar = Path("studio/ui/foundation_sidebar.py").read_text(encoding="utf-8")
    for source in (store, config, sidebar):
        assert "/lilab_data/" not in source
    assert "default_local_embedding_model" in store
    assert "default_local_embedding_model" in config
    assert "default_local_embedding_model" in sidebar


def test_worker_forces_offline_and_refuses_hf_backend():
    worker = Path("medical_kb_worker.py").read_text(encoding="utf-8")
    jobs = Path("foundation_model/rag/build_jobs.py").read_text(encoding="utf-8")
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in worker
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in worker
    assert "后台拒绝在线 Embedding 后端" in worker
    assert "建库只允许使用服务器本地 Embedding 模型" in jobs


def test_local_encoder_recovers_model_identity_from_modelscope_snapshot():
    embeddings = Path("foundation_model/rag/embeddings.py").read_text(encoding="utf-8")
    assert 'candidate_name.lower().endswith("bge-small-en-v1.5")' in embeddings
    assert 'source_name = "BAAI/bge-small-en-v1.5"' in embeddings
