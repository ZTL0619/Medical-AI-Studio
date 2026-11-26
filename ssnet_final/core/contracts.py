"""Stable domain contracts shared by the 3-D segmentation UI, training and inference layers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    task: str
    input_rank: int
    num_classes: int
    supports_semi_supervision: bool
    supports_feature_learning: bool
    available: bool
    description: str
