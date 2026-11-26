"""Simple, auditable JSONL schema for multimodal SFT and benchmark samples."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VQASample:
    image: Path
    question: str
    answer: str
    system: str = "You are a medical visual assistant. Answer carefully and concisely."
    sample_id: Optional[str] = None
