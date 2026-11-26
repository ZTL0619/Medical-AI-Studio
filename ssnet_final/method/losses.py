import contextlib

import numpy as np
import torch
import torch.nn.functional as F


def binary_dice_loss(probability, target, eps=1e-8):
    """Original studio Dice loss kept for non-DNP objectives."""
    target = target.float()
    intersection = 2 * torch.sum(probability * target) + eps
    union = torch.sum(probability) + torch.sum(target) + eps
    return 1 - intersection / union


def dnp_dice_loss(probability, target):
    """Dice loss used by the CCAI aux4 DNP training script."""
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(probability * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(probability * probability)
    return 1.0 - (2.0 * intersect + smooth) / (z_sum + y_sum + smooth)


def multiclass_dice_loss(probabilities, labels, num_classes):
    target = F.one_hot(labels.long(), num_classes=num_classes).movedim(-1, 1).float()
    total = probabilities.new_tensor(0.0)
    for class_index in range(num_classes):
        score = probabilities[:, class_index]
        truth = target[:, class_index]
        smooth = 1e-10
        intersect = torch.sum(score * truth)
        y_sum = torch.sum(truth * truth)
        z_sum = torch.sum(score * score)
        total = total + 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return total / num_classes


def softmax_mse_loss(input_values, target_values):
    input_softmax = F.softmax(input_values, dim=1)
    target_softmax = F.softmax(target_values, dim=1)
    return (input_softmax - target_softmax) ** 2


def sigmoid_rampup(current, rampup_length):
    # Exact CCAI utils/ramps.py behavior.
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def sharpen(probabilities, temperature):
    power = 1.0 / temperature
    numerator = probabilities ** power
    denominator = numerator + (1 - probabilities) ** power
    return numerator / denominator


def neural_process_cross_entropy(logits, targets_onehot, samples, mask=None):
    """Neural-Process CE matching ``ce_loss_np`` in aux4.

    ``mask=None`` is the labeled ``sup_loss``.  A [N,1] mask produces the
    masked unlabeled ``unsup_loss`` used by the selected weighted objective.
    """
    probabilities = F.softmax(logits, dim=-1)
    expanded_targets = targets_onehot.unsqueeze(0).expand(samples, -1, -1)
    batch = probabilities.size(1)

    if mask is None:
        loss = -expanded_targets * probabilities.log()
        return torch.sum(loss) / (batch * samples)

    expanded_mask = mask.unsqueeze(0).expand(samples, -1, -1)
    loss = expanded_mask * (-expanded_targets * probabilities.log())
    return torch.sum(loss) / (torch.sum(expanded_mask) + 1e-16)


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    """Match the VAT helper used by the CCAI code without updating BN stats."""
    toggled = []
    for module in model.modules():
        if hasattr(module, "track_running_stats"):
            toggled.append(module)
            module.track_running_stats ^= True
    try:
        yield
    finally:
        for module in toggled:
            module.track_running_stats ^= True


def _l2_normalize_3d(direction):
    reshaped = direction.reshape(direction.shape[0], -1, *(1 for _ in range(direction.dim() - 2)))
    return direction / (torch.norm(reshaped, dim=1, keepdim=True) + 1e-8)


def _primary_segmentation_logits(model_output):
    """Return decoder-1 logits from either the modular or original-style output."""
    if isinstance(model_output, dict):
        branches = model_output.get("branches")
        if not branches:
            raise RuntimeError("VAT requires segmentation branches in model output.")
        return branches[0]
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def vat3d_loss(model, x, xi=10.0, epi=10.0, ip=1):
    """CCAI aux4 ``VAT3d`` behavior.

    This intentionally follows the author code: create a random direction,
    call ``adv_distance.backward()`` to obtain ``d.grad``, normalize it, and
    clear model gradients with ``model.zero_grad()`` before forming the final
    LDS loss.
    """
    with torch.no_grad():
        pred = F.softmax(_primary_segmentation_logits(model(x)), dim=1)

    # CCAI: d = torch.rand(x.shape).sub(0.5).to(x.device)
    direction = torch.rand(x.shape).sub(0.5).to(x.device)
    direction = _l2_normalize_3d(direction)

    with _disable_tracking_bn_stats(model):
        for _ in range(int(ip)):
            direction.requires_grad_(True)
            pred_hat = _primary_segmentation_logits(model(x + float(xi) * direction))
            p_hat = F.softmax(pred_hat, dim=1)
            adv_distance = binary_dice_loss(p_hat, pred)
            # Match CCAI VAT3d exactly instead of autograd.grad.
            adv_distance.backward()
            direction = _l2_normalize_3d(direction.grad)
            model.zero_grad()

        pred_hat = _primary_segmentation_logits(model(x + float(epi) * direction))
        p_hat = F.softmax(pred_hat, dim=1)
        lds = binary_dice_loss(p_hat, pred)

    return lds
