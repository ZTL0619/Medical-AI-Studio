"""Command construction for single-node enterprise multi-GPU training."""
import os
import sys


def accelerate_launch_command(config, config_path):
    config.validate()
    world_size = len(config.gpu_ids)
    command = [sys.executable, "-m", "accelerate.commands.launch"]
    if world_size > 1:
        command += ["--multi_gpu", "--num_processes", str(world_size)]
    else:
        command += ["--num_processes", "1"]
    command += ["foundation_worker.py", "--config", str(config_path)]
    return command


def launch_environment(config):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in config.gpu_ids)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    return env
