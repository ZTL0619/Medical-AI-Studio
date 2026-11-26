"""Composable training objective for the 3-D integration.

For DNP the selected aux4 objective is:

    0.5 * (loss_seg_dice + loss_seg + loss_consist)
    + consistency_weight * (
        loss_lds
        + 0.05 * (loss_contr_labeled + loss_contr_unlabeled)
        + 0.01 * skew_uncertain_loss
        + 0.1 * sup_loss
    )

``unsup_loss`` is still computed/logged for diagnostics, but it is not part
of this selected total-loss expression.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from ssnet_final.method.contrastive import (
    FeatureMemory,
    contrastive_class_to_class_learned_memory,
)
from ssnet_final.method.feature_learning import similarity_fuzziness_loss
from ssnet_final.method.losses import (
    binary_dice_loss,
    neural_process_cross_entropy,
    sharpen,
    sigmoid_rampup,
    softmax_mse_loss,
    vat3d_loss,
)
from ssnet_final.method.neural_process import (
    forward_neural_process,
    released_wasserstein_distance,
)


class ModularObjective(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prototype_memory = (
            FeatureMemory(elements_per_class=32, n_classes=config.num_classes)
            if config.semi_supervision == "dnp"
            else None
        )

    def forward(self, model, images, labels, iteration):
        output = model(images)
        parts = {}

        if self.config.model == "catnet3d":
            base_loss = self._catnet_supervised(output, labels)
            parts["segmentation"] = base_loss[0].detach()
            parts["anchor"] = base_loss[1].detach()
            total = base_loss[0] + base_loss[1]
        else:
            ce, dice = self._vnet_supervised(output["branches"], labels)
            parts["segmentation_ce"] = ce.detach()
            parts["segmentation_dice"] = dice.detach()

            if self.config.semi_supervision == "dnp":
                # loss_consist is computed below and belongs inside this 0.5 term.
                total = self.config.dnp_segmentation_weight * (ce + dice)
                parts["dnp_supervised_segmentation_weight"] = ce.new_tensor(
                    self.config.dnp_segmentation_weight
                )
            else:
                # Two branches are averaged for supervised / traditional training.
                total = 0.5 * (ce + dice)

        if self.config.semi_supervision == "traditional":
            mutual = self._traditional_mutual(output["branches"], iteration)
            total = total + 0.5 * mutual
            parts["semi_traditional"] = mutual.detach()

        elif self.config.semi_supervision == "dnp":
            mutual, masks = self._dnp_mutual(output["branches"], iteration)

            contrastive_labeled, contrastive_unlabeled = self._dnp_contrastive(
                model, output, labels
            )
            np_supervised, np_unsupervised, distribution = self._dnp_distribution(
                model, output, labels, masks
            )
            loss_lds = vat3d_loss(
                model,
                images,
                xi=10.0,
                epi=self.config.dnp_vat_magnitude,
                ip=1,
            )

            # aux4 increments iter_num immediately before computing this weight.
            consistency_weight = self.config.dnp_consistency * sigmoid_rampup(
                (int(iteration) + 1) // 150,
                self.config.dnp_consistency_rampup,
            )
            consistency_weight_tensor = ce.new_tensor(consistency_weight)

            # Requested formula:
            # 0.5*(Dice + CE + loss_consist) + w(t)*(
            #   loss_lds + 0.05*(CL_l+CL_u) + 0.01*WSD + 0.1*NP_sup)
            total = total + self.config.dnp_segmentation_weight * mutual
            contrastive = contrastive_labeled + contrastive_unlabeled
            dnp_auxiliary = (
                self.config.dnp_lds_weight * loss_lds
                + self.config.dnp_contrastive_weight * contrastive
                + self.config.dnp_distribution_weight * distribution
                + self.config.dnp_np_supervised_weight * np_supervised
            )
            weighted_auxiliary = consistency_weight_tensor * dnp_auxiliary
            total = total + weighted_auxiliary

            # Keep every term visible in history/TensorBoard. NP unsup is
            # calculated for diagnostics but intentionally excluded from total.
            parts["semi_dnp"] = mutual.detach()
            parts["dnp_loss_lds"] = loss_lds.detach()
            parts["dnp_lds_weight"] = ce.new_tensor(self.config.dnp_lds_weight)
            parts["dnp_contrastive_labeled"] = contrastive_labeled.detach()
            parts["dnp_contrastive_unlabeled"] = contrastive_unlabeled.detach()
            parts["dnp_contrastive_weight"] = ce.new_tensor(self.config.dnp_contrastive_weight)
            parts["dnp_np_unsupervised"] = np_unsupervised.detach()
            parts["dnp_np_supervised"] = np_supervised.detach()
            parts["dnp_np_supervised_weight"] = ce.new_tensor(self.config.dnp_np_supervised_weight)
            parts["dnp_distribution"] = distribution.detach()
            parts["dnp_distribution_weight"] = ce.new_tensor(self.config.dnp_distribution_weight)
            parts["dnp_consistency_weight"] = consistency_weight_tensor.detach()
            parts["dnp_auxiliary_unweighted"] = dnp_auxiliary.detach()
            parts["dnp_auxiliary_weighted"] = weighted_auxiliary.detach()

            # Backward-compatible names used by earlier studio builds.
            parts["loss_lds"] = loss_lds.detach()
            parts["np_supervised"] = np_supervised.detach()
            parts["np_unsupervised"] = np_unsupervised.detach()
            parts["distribution"] = distribution.detach()

        if self.config.feature_learning == "fsl":
            original = output.get("fsl_original")
            transformed = output.get("fsl_transformed")
            if original is None or transformed is None:
                raise RuntimeError(
                    "FSL is enabled but the selected backbone did not expose transformed features."
                )
            fsl = similarity_fuzziness_loss(original, transformed, self.config.beta)
            ramp = sigmoid_rampup(iteration // 150, 40.0)
            weighted_fsl = ramp * self.config.fuzziness_weight * fsl
            total = total + weighted_fsl
            parts["fsl"] = fsl.detach()
            parts["fsl_weighted"] = weighted_fsl.detach()

        parts["total"] = total.detach()
        return total, parts

    def _catnet_supervised(self, output, labels):
        prediction = output["branches"][0]
        anchor_logits = output["anchor_logits"]
        pad_left, pad_right = output.get("depth_padding", (0, 0))
        target = labels.float()
        if pad_left or pad_right:
            target = F.pad(target, (pad_left, pad_right, 0, 0))

        pred = prediction.squeeze(1)
        smooth = 1.0
        reduce_dims = tuple(range(1, pred.dim()))
        numerator = 2.0 * torch.sum(pred * target, dim=reduce_dims)
        denominator = (
            torch.sum(pred.pow(2), dim=reduce_dims)
            + torch.sum(target.pow(2), dim=reduce_dims)
            + smooth
        )
        segmentation_loss = torch.clamp((1.0 - numerator / denominator).mean(), 0.0, 1.0)

        anchor_target = F.max_pool2d(target, kernel_size=16, stride=16).long()
        anchor_loss = F.cross_entropy(anchor_logits, anchor_target)
        return segmentation_loss, anchor_loss

    def _vnet_supervised(self, branches, labels):
        truth = labels[: self.config.labeled_batch_size]
        ce = truth.new_tensor(0.0, dtype=torch.float32)
        dice = truth.new_tensor(0.0, dtype=torch.float32)
        for branch in branches:
            branch = branch[: self.config.labeled_batch_size]
            probabilities = F.softmax(branch, dim=1)
            ce = ce + F.cross_entropy(branch, truth)
            # aux4 uses utils.losses.Binary_dice_loss for DNP as well:
            # 1 - (2 * sum(p*y) + 1e-8) / (sum(p) + sum(y) + 1e-8).
            # Do NOT replace this with the squared-denominator soft-Dice variant.
            dice = dice + binary_dice_loss(probabilities[:, 1], truth == 1)
        return ce, dice

    def _traditional_mutual(self, branches, iteration):
        unlabeled = [
            F.softmax(branch[self.config.labeled_batch_size :], dim=1)
            for branch in branches
        ]
        pseudo = [
            sharpen(probability, self.config.sharpening_temperature).detach()
            for probability in unlabeled
        ]
        threshold = (
            0.75 + 0.25 * sigmoid_rampup(iteration, self.config.max_iterations)
        ) * math.log(2)
        masks = []
        for probability in unlabeled:
            entropy = -torch.sum(
                probability * torch.log(probability + 1e-6), dim=1, keepdim=True
            )
            masks.append((entropy < threshold).float())

        total = unlabeled[0].new_tensor(0.0)
        for receiver in range(2):
            provider = 1 - receiver
            distance = softmax_mse_loss(unlabeled[receiver], pseudo[provider])
            total = total + torch.sum(masks[provider] * distance) / (
                2.0 * torch.sum(masks[provider]) + 1e-16
            )
        return total

    def _dnp_mutual(self, branches, iteration):
        probabilities = [
            F.softmax(branch[self.config.labeled_batch_size :], dim=1)
            for branch in branches
        ]
        pseudo = [
            sharpen(probability, self.config.sharpening_temperature).detach()
            for probability in probabilities
        ]
        threshold = (
            0.75 + 0.25 * sigmoid_rampup(iteration, self.config.max_iterations)
        ) * math.log(2)
        masks = []
        for probability in probabilities:
            entropy = -torch.sum(
                probability * torch.log(probability + 1e-6), dim=1, keepdim=True
            )
            masks.append((entropy < threshold).float())

        total = probabilities[0].new_tensor(0.0)
        for receiver in range(2):
            provider = 1 - receiver
            distance = softmax_mse_loss(probabilities[receiver], pseudo[provider])
            provider_matrix = masks[provider] * distance
            weighted = (
                self.config.tau * masks[receiver] * provider_matrix
                + (1.0 - masks[receiver]) * provider_matrix
            )
            total = total + torch.sum(weighted) / (
                2.0 * torch.sum(masks[provider]) + 1e-16
            )
        return total, masks

    def _dnp_contrastive(self, model, output, labels):
        if model.dnp_head is None:
            raise RuntimeError("DNP semi-supervision is enabled but the DNP head is missing.")
        if not output["embeddings"]:
            raise RuntimeError("DNP contrastive learning requires decoder embeddings.")
        if self.prototype_memory is None:
            raise RuntimeError("DNP prototype memory was not initialized.")

        head = model.dnp_head
        loss_labeled = output["branches"][0].new_tensor(0.0)
        loss_unlabeled = output["branches"][0].new_tensor(0.0)
        true_labels = labels[: self.config.labeled_batch_size]

        for branch_index, (branch_logits, embedding) in enumerate(
            zip(output["branches"], output["embeddings"])
        ):
            outputs_soft = F.softmax(branch_logits, dim=1)
            labeled_features = embedding[: self.config.labeled_batch_size]
            unlabeled_features = embedding[self.config.labeled_batch_size :]
            y = outputs_soft[: self.config.labeled_batch_size]

            prediction_label = torch.argmax(y, dim=1)
            pseudo_label = torch.argmax(
                outputs_soft[self.config.labeled_batch_size :], dim=1
            )

            correct_foreground = (
                (prediction_label == true_labels).float()
                * (prediction_label > 0).float()
            ).bool()

            labeled_features_last = labeled_features.permute(0, 2, 3, 4, 1)
            labels_correct = true_labels[correct_foreground]
            labeled_features_correct = labeled_features_last[correct_foreground, ...]

            # The reference temporarily switches the model to eval for this
            # no-grad projection so BatchNorm uses stored statistics.
            if labeled_features_correct.shape[0] > 0:
                projection_was_training = head.projection_head.training
                head.projection_head.eval()
                with torch.no_grad():
                    projected_correct = head.projection_head(labeled_features_correct)
                if projection_was_training:
                    head.projection_head.train()
                self.prototype_memory.add_features_from_sample_learned(
                    head, projected_correct, labels_correct
                )

            labeled_features_all = labeled_features_last.reshape(
                -1, labeled_features_last.size(-1)
            )
            labeled_labels = true_labels.reshape(-1)
            projected_labeled = head.projection_head(labeled_features_all)
            predicted_labeled = head.prediction_head(projected_labeled)
            loss_labeled = loss_labeled + contrastive_class_to_class_learned_memory(
                head,
                predicted_labeled,
                labeled_labels,
                self.config.num_classes,
                self.prototype_memory.memory,
            )

            unlabeled_features_all = unlabeled_features.permute(0, 2, 3, 4, 1).reshape(
                -1, labeled_features_last.size(-1)
            )
            pseudo_label_flat = pseudo_label.reshape(-1)
            projected_unlabeled = head.projection_head(unlabeled_features_all)
            predicted_unlabeled = head.prediction_head(projected_unlabeled)
            loss_unlabeled = loss_unlabeled + contrastive_class_to_class_learned_memory(
                head,
                predicted_unlabeled,
                pseudo_label_flat,
                self.config.num_classes,
                self.prototype_memory.memory,
            )

        return loss_labeled, loss_unlabeled

    def _dnp_distribution(self, model, output, labels, masks):
        if model.dnp_head is None:
            raise RuntimeError(
                "DNP semi-supervision is enabled but the Neural Process head is missing."
            )
        if not output["embeddings"]:
            raise RuntimeError("DNP requires decoder embeddings from a dual-branch V-Net.")

        # aux4 runs Neural Process only for decoder/embedding branch 0.
        feature = output["embeddings"][0]
        labeled_feature = self._flatten_feature(
            feature[: self.config.labeled_batch_size]
        )
        unlabeled_feature = self._flatten_feature(
            feature[self.config.labeled_batch_size :]
        )
        labeled_labels = labels[: self.config.labeled_batch_size].reshape(-1)
        labeled_onehot = F.one_hot(
            labeled_labels.long(), num_classes=self.config.num_classes
        ).float()

        branch0_probability = F.softmax(
            output["branches"][0][self.config.labeled_batch_size :], dim=1
        )
        pseudo_soft_branch0 = self._flatten_probability(branch0_probability)

        logits_labeled, mean_labeled, sigma_labeled = forward_neural_process(
            model.dnp_head,
            labeled_feature,
            labeled_onehot,
            labeled_feature,
            labeled_onehot,
            self.config.np_samples,
        )
        logits_unlabeled, mean_pseudo, sigma_pseudo = forward_neural_process(
            model.dnp_head,
            unlabeled_feature,
            pseudo_soft_branch0,
            labeled_feature,
            labeled_onehot,
            self.config.np_samples,
        )

        np_supervised = neural_process_cross_entropy(
            logits_labeled,
            labeled_onehot,
            self.config.np_samples,
            mask=None,
        )

        # aux4 supervises branch-0 NP unlabeled predictions with hard pseudo
        # labels and the confidence mask from the *other* decoder (branch 1).
        branch1_probability = F.softmax(
            output["branches"][1][self.config.labeled_batch_size :], dim=1
        )
        pseudo_other = torch.argmax(branch1_probability, dim=1).reshape(-1)
        pseudo_other_onehot = F.one_hot(
            pseudo_other.long(), num_classes=self.config.num_classes
        ).float()
        other_mask = self._flatten_mask(masks[1])
        np_unsupervised = neural_process_cross_entropy(
            logits_unlabeled,
            pseudo_other_onehot,
            self.config.np_samples,
            mask=other_mask,
        )

        distribution = released_wasserstein_distance(
            mean_labeled, sigma_labeled, mean_pseudo, sigma_pseudo
        )
        return np_supervised, np_unsupervised, distribution

    @staticmethod
    def _flatten_feature(feature):
        order = [0] + list(range(2, feature.dim())) + [1]
        return feature.permute(*order).reshape(-1, feature.size(1))

    @staticmethod
    def _flatten_probability(probability):
        order = [0] + list(range(2, probability.dim())) + [1]
        return probability.permute(*order).reshape(-1, probability.size(1))

    @staticmethod
    def _flatten_mask(mask):
        order = [0] + list(range(2, mask.dim())) + [1]
        return mask.permute(*order).reshape(-1, mask.size(1))


def build_objective(config):
    return ModularObjective(config)
