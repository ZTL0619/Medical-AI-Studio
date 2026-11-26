import torch

from ssnet_final.method.losses import binary_dice_loss


def test_aux4_binary_dice_formula():
    p = torch.tensor([0.2, 0.8, 0.7, 0.1], dtype=torch.float32)
    y = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    eps = 1e-8
    expected = 1.0 - (2.0 * torch.sum(p * y) + eps) / (torch.sum(p) + torch.sum(y) + eps)
    actual = binary_dice_loss(p, y, eps=eps)
    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_aux4_dice_is_not_squared_denominator_variant():
    p = torch.tensor([0.2, 0.8, 0.7, 0.1], dtype=torch.float32)
    y = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    aux4 = binary_dice_loss(p, y)
    squared = 1.0 - (2.0 * torch.sum(p*y) + 1e-5) / (torch.sum(p*p) + torch.sum(y*y) + 1e-5)
    assert not torch.allclose(aux4, squared)
