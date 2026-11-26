"""Shared 3-D dual-branch V-Net backbone.

The encoder and the two decoder branches are the structures used by the
semi-supervised reference implementations.  Feature learning and Neural
Process regularization are deliberately kept outside this file so they can be
selected independently from the backbone.
"""

import torch
from torch import nn


def normalization_layer(kind, channels):
    if kind == "batchnorm":
        return nn.BatchNorm3d(channels)
    if kind == "groupnorm":
        return nn.GroupNorm(num_groups=16, num_channels=channels)
    if kind == "instancenorm":
        return nn.InstanceNorm3d(channels)
    if kind == "none":
        return None
    raise ValueError(kind)


class ConvBlock(nn.Module):
    def __init__(self, depth, in_channels, out_channels, normalization):
        super().__init__()
        layers = []
        for index in range(depth):
            current_in = in_channels if index == 0 else out_channels
            layers.append(nn.Conv3d(current_in, out_channels, 3, padding=1))
            norm = normalization_layer(normalization, out_channels)
            if norm is not None:
                layers.append(norm)
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization):
        super().__init__()
        layers = [nn.Conv3d(in_channels, out_channels, 2, stride=2)]
        norm = normalization_layer(normalization, out_channels)
        if norm is not None:
            layers.append(norm)
        layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization, mode):
        super().__init__()
        if mode == 0:
            layers = [nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)]
        elif mode == 1:
            layers = [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv3d(in_channels, out_channels, 3, padding=1),
            ]
        else:
            raise ValueError(mode)
        norm = normalization_layer(normalization, out_channels)
        if norm is not None:
            layers.append(norm)
        layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, normalization, has_dropout):
        super().__init__()
        self.block_one = ConvBlock(1, in_channels, base_channels, normalization)
        self.block_one_dw = DownBlock(base_channels, base_channels * 2, normalization)
        self.block_two = ConvBlock(2, base_channels * 2, base_channels * 2, normalization)
        self.block_two_dw = DownBlock(base_channels * 2, base_channels * 4, normalization)
        self.block_three = ConvBlock(3, base_channels * 4, base_channels * 4, normalization)
        self.block_three_dw = DownBlock(base_channels * 4, base_channels * 8, normalization)
        self.block_four = ConvBlock(3, base_channels * 8, base_channels * 8, normalization)
        self.block_four_dw = DownBlock(base_channels * 8, base_channels * 16, normalization)
        self.block_five = ConvBlock(3, base_channels * 16, base_channels * 16, normalization)
        self.dropout = nn.Dropout3d(0.5)
        self.has_dropout = has_dropout

    def forward(self, x):
        x1 = self.block_one(x)
        x2 = self.block_two(self.block_one_dw(x1))
        x3 = self.block_three(self.block_two_dw(x2))
        x4 = self.block_four(self.block_three_dw(x3))
        x5 = self.block_five(self.block_four_dw(x4))
        if self.has_dropout:
            x5 = self.dropout(x5)
        return [x1, x2, x3, x4, x5]


class Decoder(nn.Module):
    def __init__(self, num_classes, base_channels, normalization, has_dropout, mode):
        super().__init__()
        self.block_five_up = UpBlock(base_channels * 16, base_channels * 8, normalization, mode)
        self.block_six = ConvBlock(3, base_channels * 8, base_channels * 8, normalization)
        self.block_six_up = UpBlock(base_channels * 8, base_channels * 4, normalization, mode)
        self.block_seven = ConvBlock(3, base_channels * 4, base_channels * 4, normalization)
        self.block_seven_up = UpBlock(base_channels * 4, base_channels * 2, normalization, mode)
        self.block_eight = ConvBlock(2, base_channels * 2, base_channels * 2, normalization)
        self.block_eight_up = UpBlock(base_channels * 2, base_channels, normalization, mode)
        self.block_nine = ConvBlock(1, base_channels, base_channels, normalization)
        self.out_conv = nn.Conv3d(base_channels, num_classes, 1)
        self.dropout = nn.Dropout3d(0.5)
        self.has_dropout = has_dropout

    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        x6 = self.block_six(self.block_five_up(x5) + x4)
        x7 = self.block_seven(self.block_six_up(x6) + x3)
        x8 = self.block_eight(self.block_seven_up(x7) + x2)
        x9 = self.block_nine(self.block_eight_up(x8) + x1)
        if self.has_dropout:
            x9 = self.dropout(x9)
        return self.out_conv(x9), x9


class VNet3D(nn.Module):
    """Dual-branch 3-D V-Net with an optional bottleneck feature module."""

    bottleneck_channels = 256
    embedding_channels = 16
    supports_semi_supervision = True

    def __init__(
        self,
        n_channels=1,
        n_classes=2,
        n_filters=16,
        normalization="batchnorm",
        has_dropout=True,
    ):
        super().__init__()
        self.encoder = Encoder(n_channels, n_filters, normalization, has_dropout)
        self.decoder_1 = Decoder(n_classes, n_filters, normalization, has_dropout, 0)
        self.decoder_2 = Decoder(n_classes, n_filters, normalization, has_dropout, 1)
        self.bottleneck_channels = n_filters * 16
        self.embedding_channels = n_filters

    def forward(self, x, feature_module=None):
        features = self.encoder(x)
        original = None
        transformed = None
        if feature_module is not None:
            features[4], original, transformed = feature_module(features[4])
        out1, embedding1 = self.decoder_1(features)
        out2, embedding2 = self.decoder_2(features)
        return {
            "branches": [out1, out2],
            "embeddings": [embedding1, embedding2],
            "anchor_logits": None,
            "fsl_original": original,
            "fsl_transformed": transformed,
            "binary_probability": False,
            "depth_padding": (0, 0),
        }
