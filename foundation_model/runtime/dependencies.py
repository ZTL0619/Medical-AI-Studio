"""Optional dependency checks that never break the segmentation application."""
from importlib.util import find_spec


FOUNDATION_PACKAGES = {
    "transformers": "Transformers model loading/inference",
    "accelerate": "multi-GPU launcher",
    "peft": "LoRA/QLoRA adapters",
    "bitsandbytes": "4-bit QLoRA",
    "datasets": "SFT dataset loading",
    "trl": "multimodal SFT trainer",
    "deepspeed": "ZeRO-2 training option",
    "vllm": "production-style model serving",
}


def dependency_status():
    return {
        name: {"installed": find_spec(name) is not None, "purpose": purpose}
        for name, purpose in FOUNDATION_PACKAGES.items()
    }


def require(packages):
    missing = [name for name in packages if find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Missing optional foundation-model dependencies: " + ", ".join(missing) +
            ". Install the appropriate project extras before running this operation."
        )
