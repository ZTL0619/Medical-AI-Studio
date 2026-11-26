"""Factories for independently selectable 3-D project modules."""

from ssnet_final.method.feature_learning import FSLFeatureLearning
from ssnet_final.method.neural_process import DNPHead
from ssnet_final.models.catnet3d import CATNet3D
from ssnet_final.models.registry import get_model_spec
from ssnet_final.models.system import Modular3DSystem
from ssnet_final.models.vnet import VNet3D


def _catnet_params(config):
    return {
        "in_chns": 1,
        "is_training": True,
        "is_anchor": True,
        "T": float(config.anchor_temperature),
        "feature_chns": [16, 32, 64, 128, 256],
        "dropout": [0, 0, 0.3, 0.4, 0.5],
        "class_num": 1,
        "bilinear": True,
        "input_resolution": [112, 112],
        "depth": [2, 2, 2, 2],
        "num_heads": [1, 4, 4, 8, 16],
        "window_size": 7,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "mlp_ratio": 4.0,
        "use_checkpoint": False,
    }


def build_backbone(config):
    spec = get_model_spec(config.model)
    if config.model == "vnet":
        return VNet3D(
            n_channels=1,
            n_classes=config.num_classes,
            normalization="batchnorm",
            has_dropout=True,
        )
    if config.model == "catnet3d":
        return CATNet3D(_catnet_params(config))
    raise ValueError(f"Unsupported model: {spec.key}")


def build_model(config, training=True):
    backbone = build_backbone(config)

    feature_learning = None
    if config.feature_learning == "fsl":
        feature_learning = FSLFeatureLearning(backbone.bottleneck_channels)

    dnp_head = None
    if config.semi_supervision == "dnp":
        if not getattr(backbone, "supports_semi_supervision", False):
            raise ValueError("DNP semi-supervision requires a dual-branch backbone; select MODEL='vnet'.")
        dnp_head = DNPHead(backbone.embedding_channels, config.num_classes)

    system = Modular3DSystem(backbone, feature_learning=feature_learning, dnp_head=dnp_head)
    system.train(bool(training))
    return system
