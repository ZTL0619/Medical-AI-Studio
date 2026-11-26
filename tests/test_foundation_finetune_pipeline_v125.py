from pathlib import Path

import torch

from foundation_model.config import FoundationTrainConfig
from foundation_model.models.registry import get_foundation_model
from foundation_model.training.trainer import _mask_assistant_only_labels


def test_registry_contains_medgemma_and_lingshu():
    assert get_foundation_model("medgemma4b").default_model_id == "google/medgemma-4b-it"
    assert get_foundation_model("lingshu7b").multimodal is True


def test_assistant_only_label_mask_masks_prompt_and_padding():
    labels = torch.tensor([[10, 11, 12, 13, 14, 0], [20, 21, 22, 23, 0, 0]])
    masked = _mask_assistant_only_labels(labels, [3, 2], pad_token_id=0)
    assert masked.tolist() == [
        [-100, -100, -100, 13, 14, -100],
        [-100, -100, 22, 23, -100, -100],
    ]


def test_validation_config_roundtrip(tmp_path):
    config = FoundationTrainConfig(
        dataset_path=tmp_path / "train.jsonl",
        validation_dataset_path=tmp_path / "val.jsonl",
        validation_split=0.2,
        output_dir=tmp_path / "out",
        gpu_ids=(0,),
    ).validate()
    restored = FoundationTrainConfig.from_json_dict(config.to_json_dict())
    assert restored.validation_dataset_path == tmp_path / "val.jsonl"
    assert restored.validation_split == 0.2


def test_on_demand_transformers_worker_is_present_and_used():
    root = Path(__file__).resolve().parents[1]
    assert (root / "foundation_infer_worker.py").exists()
    assert not (root / "foundation_transformers_service.py").exists()
    runtime = (root / "studio" / "foundation_runtime.py").read_text(encoding="utf-8")
    page = (root / "studio" / "foundation_page.py").read_text(encoding="utf-8")
    assert "foundation_infer_worker.py" in runtime
    assert "subprocess.run(" in runtime
    assert "transformers_manager.run(" in page
    assert "run_resident" not in runtime
