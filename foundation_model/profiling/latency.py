"""Inference profiling records."""
from dataclasses import asdict, dataclass


@dataclass
class InferenceProfile:
    total_latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    peak_vram_mb: float = 0.0

    def to_dict(self):
        return asdict(self)
