import torch

from ssnet_final.config import build_config
from ssnet_final.method.objective import build_objective
from ssnet_final.models.factory import build_model


def test_vnet_fsl_dnp_forward_backward_cpu():
    config = build_config(
        command="train",
        dataset="LA",
        model="vnet",
        feature_learning="fsl",
        semi_supervision="dnp",
        labeled_percent=10,
        data_root="/tmp/la",
        output="/tmp/out",
        device="cpu",
        batch_size=4,
        labeled_batch_size=2,
        max_iterations=10,
    )
    model = build_model(config, training=True)
    objective = build_objective(config)
    image = torch.randn(4, 1, 16, 16, 16)
    label = torch.randint(0, 2, (4, 16, 16, 16))
    loss, _ = objective(model, image, label, 1)
    loss.backward()
    assert model.feature_learning.transform.weight.grad is not None
    assert model.dnp_head.classifier.weight.grad is not None


def test_dnp_weighted_aux4_total_matches_selected_formula_cpu():
    config = build_config(
        command="train",
        dataset="LA",
        model="vnet",
        feature_learning="none",
        semi_supervision="dnp",
        labeled_percent=5,
        data_root="/tmp/la",
        output="/tmp/out",
        device="cpu",
        batch_size=4,
        labeled_batch_size=2,
        max_iterations=15000,
    )
    model = build_model(config, training=True)
    objective = build_objective(config)
    image = torch.randn(4, 1, 16, 16, 16)
    label = torch.randint(0, 2, (4, 16, 16, 16))
    loss, parts = objective(model, image, label, 0)

    expected = 0.5 * (
        parts["segmentation_dice"]
        + parts["segmentation_ce"]
        + parts["semi_dnp"]
    )
    expected = expected + parts["dnp_consistency_weight"] * (
        parts["dnp_loss_lds"]
        + 0.05 * (
            parts["dnp_contrastive_labeled"]
            + parts["dnp_contrastive_unlabeled"]
        )
        + 0.01 * parts["dnp_distribution"]
        + 0.1 * parts["dnp_np_supervised"]
    )
    assert torch.allclose(loss.detach(), expected, atol=1e-6, rtol=1e-6)
    assert config.dnp_segmentation_weight == 0.5
    assert config.dnp_contrastive_weight == 0.05
    assert config.dnp_consistency == 1.0
    assert config.dnp_consistency_rampup == 40.0
    assert config.dnp_lds_weight == 1.0
    assert config.dnp_distribution_weight == 0.01
    assert config.dnp_np_supervised_weight == 0.1
    assert config.dnp_vat_magnitude == 10.0
