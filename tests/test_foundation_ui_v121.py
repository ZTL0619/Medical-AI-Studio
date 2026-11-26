from pathlib import Path

from foundation_model.config import FoundationServingConfig, FoundationTrainConfig
from foundation_model.distributed.launcher import accelerate_launch_command, launch_environment
from foundation_model.serving.vllm_manager import VLLMServiceManager
from studio.foundation_ui_state import is_foundation_training_running


def test_training_running_helper_never_returns_none():
    assert is_foundation_training_running(None) is False
    assert is_foundation_training_running({}) is False
    assert is_foundation_training_running({"process_alive": None}) is False
    assert is_foundation_training_running({"state": "running", "process_alive": None}) is True
    assert isinstance(is_foundation_training_running({"state": "completed"}), bool)


def test_foundation_ui_is_single_chat_workspace_not_developer_tabs():
    page = Path("studio/foundation_page.py").read_text(encoding="utf-8")
    sidebar = Path("studio/ui/foundation_sidebar.py").read_text(encoding="utf-8")
    assert 'st.tabs(' not in page
    assert 'st.chat_input(' in page
    assert '医疗多模态助手' in page
    assert '启用模型微调' in sidebar
    assert '启用医学知识增强' in sidebar
    assert '多 GPU 推理策略' in sidebar
    assert '双 GPU' not in sidebar
    assert '双GPU' not in sidebar


def test_multi_gpu_training_scales_to_arbitrary_world_size(tmp_path):
    config = FoundationTrainConfig(
        dataset_path=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        gpu_ids=(0, 2, 3, 4),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
    ).validate()
    assert config.world_size == 4
    assert config.global_batch_size == 32
    command = accelerate_launch_command(config, tmp_path / "config.json")
    assert command[command.index("--num_processes") + 1] == "4"
    assert launch_environment(config)["CUDA_VISIBLE_DEVICES"] == "0,2,3,4"


def test_vllm_multi_gpu_data_parallel_is_not_fixed_to_two(monkeypatch, tmp_path):
    monkeypatch.setattr("foundation_model.serving.vllm_manager.shutil.which", lambda _: "/usr/bin/vllm")
    manager = VLLMServiceManager(tmp_path)
    config = FoundationServingConfig(base_model="base", gpu_ids=(0, 1, 2), strategy="data_parallel")
    command = manager.build_command(config)
    assert command[command.index("--data-parallel-size") + 1] == "3"
