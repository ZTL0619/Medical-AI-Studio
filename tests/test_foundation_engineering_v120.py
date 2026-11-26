from pathlib import Path

import pytest

from foundation_model.config import FoundationServingConfig, FoundationTrainConfig
from foundation_model.data.jsonl_dataset import validate_jsonl_dataset
from foundation_model.distributed.deepspeed import zero2_config
from foundation_model.distributed.launcher import accelerate_launch_command, launch_environment
from foundation_model.rag.null_backend import KnowledgeBaseNotConfigured
from foundation_model.rag.pipeline import HybridGraphRAGPipeline
from foundation_model.serving.vllm_manager import VLLMServiceManager


def test_two_gpu_qlora_global_batch_and_accelerate_command(tmp_path):
    config = FoundationTrainConfig(
        dataset_path=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        gpu_ids=(2, 4),
        distributed_strategy="ddp",
        peft_strategy="qlora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
    ).validate()
    assert config.world_size == 2
    assert config.global_batch_size == 32
    command = accelerate_launch_command(config, tmp_path / "config.json")
    assert "accelerate.commands.launch" in command
    assert "--multi_gpu" in command
    assert command[command.index("--num_processes") + 1] == "2"
    assert launch_environment(config)["CUDA_VISIBLE_DEVICES"] == "2,4"


def test_deepspeed_zero2_contract():
    config = FoundationTrainConfig(gpu_ids=(0, 1), distributed_strategy="deepspeed_zero2")
    payload = zero2_config(config)
    assert payload["zero_optimization"]["stage"] == 2
    assert payload["gradient_accumulation_steps"] == config.gradient_accumulation_steps


def test_vllm_two_gpu_data_parallel_command(monkeypatch, tmp_path):
    monkeypatch.setattr("foundation_model.serving.vllm_manager.shutil.which", lambda _: "/usr/bin/vllm")
    manager = VLLMServiceManager(tmp_path)
    config = FoundationServingConfig(
        base_model="google/medgemma-4b-it",
        gpu_ids=(0, 1),
        strategy="data_parallel",
        precision="bf16",
    )
    command = manager.build_command(config)
    assert command[:2] == ["/usr/bin/vllm", "serve"]
    assert command[command.index("--data-parallel-size") + 1] == "2"
    assert "--tensor-parallel-size" not in command


def test_vllm_int4_and_lora_adapter_flags(monkeypatch, tmp_path):
    monkeypatch.setattr("foundation_model.serving.vllm_manager.shutil.which", lambda _: "/usr/bin/vllm")
    manager = VLLMServiceManager(tmp_path)
    config = FoundationServingConfig(
        base_model="base",
        adapter_path=Path("/tmp/adapter"),
        gpu_ids=(0, 1),
        strategy="tensor_parallel",
        precision="int4",
    )
    command = manager.build_command(config)
    assert "--quantization" in command and "bitsandbytes" in command
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert "--enable-lora" in command
    assert "medical-adapter=/tmp/adapter" in command


def test_graph_rag_interface_is_present_but_knowledge_base_is_deferred():
    pipeline = HybridGraphRAGPipeline()
    assert pipeline.ready is False
    with pytest.raises(KnowledgeBaseNotConfigured):
        pipeline.retrieve("left atrium MRI", top_k=5, graph_hops=1)


def test_bundled_foundation_demo_dataset_is_valid():
    root = Path(__file__).resolve().parents[1]
    report = validate_jsonl_dataset(root / "examples" / "foundation_dataset_sample.jsonl")
    assert report["valid"] is True
    assert report["samples"] == 1
