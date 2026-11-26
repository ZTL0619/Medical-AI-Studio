import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0


def initialize_distributed(config):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1

    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP multi-GPU training requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but no CUDA device is available.")

    return DistributedContext(enabled, rank, local_rank, world_size, device)


def cleanup_distributed(context):
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context):
    if context.enabled:
        dist.barrier()


def reduce_mean_dict(values, context):
    if not context.enabled:
        return {key: float(value.detach().item()) for key, value in values.items()}
    reduced = {}
    for key, value in values.items():
        tensor = value.detach().float().to(context.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= context.world_size
        reduced[key] = float(tensor.item())
    return reduced


def broadcast_float(value, context, src=0):
    tensor = torch.tensor(float(value), dtype=torch.float32, device=context.device)
    if context.enabled:
        dist.broadcast(tensor, src=src)
    return float(tensor.item())
