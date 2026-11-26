"""Registry of supported medical multimodal foundation models."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FoundationModelSpec:
    key: str
    display_name: str
    default_model_id: str
    multimodal: bool
    task: str
    description: str


_MODEL_SPECS = {
    "medgemma4b": FoundationModelSpec(
        key="medgemma4b",
        display_name="MedGemma-4B",
        default_model_id="google/medgemma-4b-it",
        multimodal=True,
        task="image_text_to_text",
        description="Google MedGemma 4B instruction-tuned medical multimodal model.",
    ),
    "lingshu7b": FoundationModelSpec(
        key="lingshu7b",
        display_name="Lingshu-7B",
        default_model_id="",
        multimodal=True,
        task="image_text_to_text",
        description="Medical multimodal VLM: image + text -> text; suitable for local inference and PEFT.",
    ),
    "custom_local_vlm": FoundationModelSpec(
        key="custom_local_vlm",
        display_name="Custom Local VLM",
        default_model_id="",
        multimodal=True,
        task="image_text_to_text",
        description="Another local/Hugging Face Transformers-compatible vision-language model.",
    ),
}


def list_foundation_models():
    return tuple(_MODEL_SPECS.values())


def get_foundation_model(key: str):
    try:
        return _MODEL_SPECS[str(key)]
    except KeyError as exc:
        raise ValueError(f"Unknown foundation model: {key}") from exc
