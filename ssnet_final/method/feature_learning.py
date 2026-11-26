"""Optional feature-learning modules."""

import torch
from torch import nn


class FSLFeatureLearning(nn.Module):
    """Learnable feature transformation Θ from the FSL reference method."""

    def __init__(self, channels):
        super().__init__()
        self.transform = nn.Linear(channels, channels)

    def forward(self, feature_map):
        if feature_map.dim() != 5:
            raise ValueError("FSLFeatureLearning expects a 3-D feature map [B,C,H,W,D]")
        batch, channels, h, w, d = feature_map.shape
        original = feature_map.permute(0, 2, 3, 4, 1).reshape(-1, channels)
        transformed = self.transform(original)
        transformed_map = transformed.reshape(batch, h, w, d, channels).permute(0, 4, 1, 2, 3).contiguous()
        return transformed_map, original, transformed


def similarity_fuzziness_loss(features, transformed_features, beta):
    """Pairwise similarity-fuzziness objective used by FSL."""

    def pairwise_distance(values):
        gram = torch.matmul(values, values.t())
        diagonal = gram.diag().unsqueeze(0).expand_as(gram)
        squared = (diagonal + diagonal.t() - 2.0 * gram).clamp_min(0.0)
        return torch.sqrt(squared + 1e-4)

    distance = pairwise_distance(features)
    rho = 1.0 / (1.0 + beta * distance)
    distance_w = pairwise_distance(transformed_features)
    rho_w = 1.0 / (1.0 + beta * distance_w)
    objective = rho_w * (1.0 - rho) + (1.0 - rho_w) * rho
    return torch.mean(objective)
