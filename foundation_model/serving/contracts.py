"""Serving contracts shared by UI, vLLM manager and benchmark runner."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ServingEndpoint:
    base_url: str
    model_name: str
    backend: str = "vllm"
