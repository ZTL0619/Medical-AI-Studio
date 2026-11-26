"""Benchmark contracts inspired by standardized medical evaluation pipelines."""
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    image: str | None
    prompt: str
    reference: str
    metadata: dict


@dataclass(frozen=True)
class BenchmarkPrediction:
    sample_id: str
    prediction: str
    reference: str
    latency_s: float
    metadata: dict


class BenchmarkAdapter(Protocol):
    def load(self) -> Sequence[BenchmarkSample]: ...


class Metric(Protocol):
    name: str
    def compute(self, predictions: Sequence[BenchmarkPrediction]) -> float: ...
