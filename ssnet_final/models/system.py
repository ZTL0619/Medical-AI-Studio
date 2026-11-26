"""Composable 3-D segmentation system."""

from torch import nn


class Modular3DSystem(nn.Module):
    def __init__(self, backbone, feature_learning=None, dnp_head=None):
        super().__init__()
        self.backbone = backbone
        self.feature_learning = feature_learning
        self.dnp_head = dnp_head

    @property
    def supports_semi_supervision(self):
        return bool(getattr(self.backbone, "supports_semi_supervision", False))

    def forward(self, x):
        return self.backbone(x, feature_module=self.feature_learning)
