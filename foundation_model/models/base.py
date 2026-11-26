"""Project-owned model adapter contract for medical image+text foundation models."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MultimodalRequest:
    image_path: str
    prompt: str
    system_prompt: str | None = None


class FoundationModelAdapter(Protocol):
    key: str
    model_id: str

    def build_messages(self, request: MultimodalRequest) -> list[dict]: ...
