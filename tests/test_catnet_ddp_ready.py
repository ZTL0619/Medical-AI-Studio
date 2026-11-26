from ssnet_final.config import build_config
from ssnet_final.models.factory import build_model


def _config():
    return build_config(
        command="train",
        dataset="LA",
        model="catnet3d",
        feature_learning="none",
        semi_supervision="none",
        labeled_percent=100,
        data_root=".",
        output="./tmp",
        device="cpu",
    )


def test_catnet_historical_inactive_branches_are_frozen():
    model = build_model(_config(), training=True)
    backbone = model.backbone

    assert all(not p.requires_grad for p in backbone.up_branch_1.parameters())
    assert all(not p.requires_grad for p in backbone.out_conv_1.parameters())

    for module in backbone.modules():
        if module.__class__.__name__ == "ConvBlock":
            assert all(not p.requires_grad for p in module.conv_1.parameters())
        if module.__class__.__name__ == "DownBlock":
            assert all(not p.requires_grad for p in module.maxpool_conv.parameters())


def test_catnet_still_has_trainable_main_path_parameters():
    model = build_model(_config(), training=True)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert any("backbone.in_conv.conv_conv" in name for name in trainable)
    assert any("backbone.up1" in name for name in trainable)
    assert any("backbone.out_conv" in name for name in trainable)
    assert any("backbone.out_conv_a" in name for name in trainable)
