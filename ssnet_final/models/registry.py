"""Central registry for 3-D segmentation model capabilities."""

from ssnet_final.core.contracts import ModelSpec


_MODEL_SPECS = {
    "vnet": ModelSpec(
        key="vnet",
        display_name="V-Net 3D",
        task="3d_segmentation",
        input_rank=5,
        num_classes=2,
        supports_semi_supervision=True,
        supports_feature_learning=True,
        available=True,
        description="3D CNN baseline; supports supervised, FSL and DNP workflows.",
    ),
    "catnet3d": ModelSpec(
        key="catnet3d",
        display_name="CAT-Net 3D",
        task="3d_segmentation",
        input_rank=5,
        num_classes=2,
        supports_semi_supervision=False,
        supports_feature_learning=False,
        available=True,
        description="3D CAT-Net supervised segmentation backbone.",
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    try:
        return _MODEL_SPECS[str(key).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown segmentation model: {key}") from exc


def list_model_specs():
    return tuple(_MODEL_SPECS.values())
