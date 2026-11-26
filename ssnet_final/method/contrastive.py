"""Class-wise contrastive learning utilities used by the DNP reference loss.

This module mirrors the auxiliary contrastive path used by
``train_ss_3d_double_cross_uncertainty_np_aux4.py``: a learned prototype
memory, a projection/prediction pair, and learned class selectors.
"""

import numpy as np
import torch
import torch.nn.functional as F


class FeatureMemory:
    """Per-class prototype memory matching the CCAI DNP implementation."""

    def __init__(self, elements_per_class=32, n_classes=2):
        self.elements_per_class = int(elements_per_class)
        self.memory = [None] * int(n_classes)
        self.n_classes = int(n_classes)

    def add_features_from_sample_learned(self, head, features, class_labels):
        features = features.detach()
        class_labels = class_labels.detach().cpu().numpy()

        for class_index in range(self.n_classes):
            mask = class_labels == class_index
            selector = getattr(head, f"contrastive_class_selector_{class_index}")
            features_class = features[mask, :]
            if features_class.shape[0] == 0:
                continue

            if features_class.shape[0] > self.elements_per_class:
                with torch.no_grad():
                    rank = torch.sigmoid(selector(features_class))
                    _, indices = torch.sort(rank[:, 0], dim=0)
                    indices = indices.cpu().numpy()
                    features_numpy = features_class.cpu().numpy()
                    # Keep the same ordering/selection as the reference code.
                    new_features = features_numpy[indices, :][: self.elements_per_class, :]
            else:
                new_features = features_class.cpu().numpy()

            self.memory[class_index] = new_features


def contrastive_class_to_class_learned_memory(head, features, class_labels, num_classes, memory):
    """Reference class-to-class learned-memory contrastive loss."""

    # Match CCAI exactly: selectors that are not used on this iteration stay
    # outside the graph instead of receiving an artificial zero gradient.
    loss = 0
    for class_index in range(int(num_classes)):
        mask = class_labels == class_index
        features_class = features[mask, :]
        memory_class = memory[class_index]

        selector = getattr(head, f"contrastive_class_selector_{class_index}")
        selector_memory = getattr(head, f"contrastive_class_selector_memory{class_index}")

        if memory_class is None or features_class.shape[0] <= 1 or memory_class.shape[0] <= 1:
            continue

        memory_tensor = torch.as_tensor(
            np.asarray(memory_class), dtype=features.dtype, device=features.device
        )
        memory_normalized = F.normalize(memory_tensor, dim=1)
        features_normalized = F.normalize(features_class, dim=1)

        similarities = torch.mm(features_normalized, memory_normalized.transpose(1, 0))
        distances = 1.0 - similarities

        learned_weights_features = torch.sigmoid(selector(features_class.detach()))
        rescaled_weights = (
            learned_weights_features.shape[0] / learned_weights_features.sum(dim=0)
        ) * learned_weights_features
        distances = distances * rescaled_weights.repeat(1, distances.shape[1])

        learned_weights_memory = torch.sigmoid(selector_memory(memory_tensor)).permute(1, 0)
        rescaled_weights_memory = (
            learned_weights_memory.shape[0] / learned_weights_memory.sum(dim=0)
        ) * learned_weights_memory
        distances = distances * rescaled_weights_memory.repeat(distances.shape[0], 1)

        loss = loss + distances.mean()

    return loss / float(num_classes)
