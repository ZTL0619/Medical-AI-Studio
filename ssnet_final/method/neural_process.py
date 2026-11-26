"""Neural Process module used by DNP semi-supervision."""

import torch
import torch.nn.functional as F
from torch import nn


class DNPHead(nn.Module):
    """Trainable Neural Process head, independent from the segmentation model."""

    def __init__(self, feature_channels, num_classes):
        super().__init__()
        # Keep module creation order identical to CCAI VNetDNP so the same
        # torch.manual_seed produces the same DNP/contrastive initialization.
        feature_dim = 32
        self.projection_head = nn.Sequential(
            nn.Linear(feature_channels, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.latent_path = nn.Sequential(
            nn.Linear(feature_channels + num_classes, 8),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.deterministic_path = nn.Sequential(
            nn.Linear(feature_channels + num_classes, 8),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.mean_net = nn.Sequential(
            nn.Linear(8, 8), nn.LeakyReLU(0.1, inplace=True), nn.Linear(8, 8)
        )
        self.log_var_net = nn.Sequential(
            nn.Linear(8, 8), nn.LeakyReLU(0.1, inplace=True), nn.Linear(8, 8)
        )
        self.fc_decoder = nn.Sequential(nn.Linear(32, 8), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(8, num_classes, bias=True)

        for class_index in range(num_classes):
            selector = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(feature_dim, 1),
            )
            setattr(self, f"contrastive_class_selector_{class_index}", selector)

        for class_index in range(num_classes):
            selector = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(feature_dim, 1),
            )
            setattr(self, f"contrastive_class_selector_memory{class_index}", selector)


def reparameterize(mean, std):
    return mean + torch.randn_like(std) * std


def forward_neural_process(head, x, labels, context_x, context_labels, samples):
    deterministic_input = torch.cat((context_x, context_labels), dim=-1)
    deterministic_representation = head.deterministic_path(deterministic_input)
    latent_input = torch.cat((x, labels), dim=-1)
    latent_representation = head.latent_path(latent_input)
    mean = head.mean_net(latent_representation.mean(0))
    raw_scale = head.log_var_net(latent_representation.mean(0))
    sigma = 0.1 + 0.9 * F.softplus(raw_scale)
    z = torch.stack([reparameterize(mean, sigma) for _ in range(samples)], dim=0)
    batch = x.size(0)
    x_expand = x.unsqueeze(0).expand(samples, -1, -1)
    context_expand = deterministic_representation.mean(0).view(1, 1, -1).expand(samples, batch, -1)
    z_expand = z.unsqueeze(1).expand(-1, batch, -1)
    decoder_input = torch.cat((z_expand, x_expand, context_expand), dim=-1)
    output = head.classifier(head.fc_decoder(decoder_input))
    return output, mean, sigma


def released_wasserstein_distance(mean_labeled, sigma_labeled, mean_pseudo, sigma_pseudo):
    mean_distance = torch.sum((mean_pseudo - mean_labeled) ** 2)
    scale_distance = torch.sum((torch.sqrt(sigma_pseudo) - torch.sqrt(sigma_labeled)) ** 2)
    return mean_distance + scale_distance
