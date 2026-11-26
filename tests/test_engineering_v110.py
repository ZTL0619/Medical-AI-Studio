from pathlib import Path

import torch

from foundation_model.config import FoundationTrainConfig, RAGConfig
from foundation_model.models.registry import get_foundation_model
from foundation_model.rag.pipeline import HybridGraphRAGPipeline
from ssnet_final.config import build_config
from ssnet_final.engine.checkpoint import load_checkpoint, save_checkpoint
from ssnet_final.models.factory import build_model
from studio.reproducibility import write_manifest


def _vnet_config(tmp_path):
    return build_config(
        command="train",
        dataset="LA",
        model="vnet",
        feature_learning="none",
        semi_supervision="none",
        labeled_percent=100,
        data_root=tmp_path / "data",
        output=tmp_path / "out",
        device="cpu",
        workers=0,
    )


def test_foundation_vlm_and_peft_contracts_are_independent_from_segmentation(tmp_path):
    spec = get_foundation_model("medgemma4b")
    assert spec.multimodal is True
    assert spec.task == "image_text_to_text"
    config = FoundationTrainConfig(
        dataset_path=tmp_path / "train.jsonl",
        output_dir=tmp_path / "fm_out",
        gpu_ids=(0, 1),
        distributed_strategy="ddp",
        peft_strategy="qlora",
    ).validate()
    assert config.world_size == 2
    assert config.global_batch_size == 16
    assert RAGConfig(enabled=False).validate().graph_hops == 1
    assert HybridGraphRAGPipeline().ready is False


def test_checkpoint_is_atomic_and_carries_schema(tmp_path):
    config = _vnet_config(tmp_path)
    model = build_model(config, training=False)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, None, iteration=12, best_score=0.8, config=config)
    assert path.exists()
    assert not list(tmp_path.glob("model.pt.*.tmp"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 2
    assert payload["model_meta"]["model"] == "vnet"
    loaded = build_model(config, training=False)
    iteration, best, _ = load_checkpoint(path, loaded, map_location="cpu", expected_config=config)
    assert iteration == 12
    assert best == 0.8


def test_reproducibility_manifest_written(tmp_path):
    config = _vnet_config(tmp_path)
    path = write_manifest(tmp_path / "out", config, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert '"schema_version": 1' in text
    assert '"torch"' in text
    assert '"experiment"' in text
