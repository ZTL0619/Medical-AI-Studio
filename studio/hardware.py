import torch


def hardware_info():
    cuda = torch.cuda.is_available()
    gpus = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "id": index,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024 ** 3), 2),
                }
            )
    return {
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "gpu_count": len(gpus),
        "gpus": gpus,
    }
