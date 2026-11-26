from pathlib import Path

import numpy as np
import torch

from ssnet_final.method.losses import sharpen, sigmoid_rampup
from ssnet_final.method.neural_process import DNPHead


def test_sigmoid_rampup_matches_caai_numpy_formula():
    for current in [0, 1, 10, 40, 100]:
        clipped = np.clip(current, 0.0, 40.0)
        phase = 1.0 - clipped / 40.0
        expected = float(np.exp(-5.0 * phase * phase))
        assert sigmoid_rampup(current, 40.0) == expected


def test_sharpen_is_unclamped_caai_formula():
    p = torch.tensor([0.2, 0.8], dtype=torch.float32)
    power = 1.0 / 0.4
    expected = p**power / (p**power + (1-p)**power)
    assert torch.equal(sharpen(p, 0.4), expected)


def test_random_crop_uses_caai_exclusive_upper_bound():
    source = Path('ssnet_final/data/la.py').read_text(encoding='utf-8')
    assert 'np.random.randint(0, current - target)' in source
    assert 'current - target + 1' not in source


def test_contrastive_has_no_artificial_zero_parameter_edges():
    source = Path('ssnet_final/method/contrastive.py').read_text(encoding='utf-8')
    assert '0.0 * sum(parameter.sum()' not in source
    assert 'features.sum() * 0.0' not in source


def test_dnp_creation_order_matches_caai():
    torch.manual_seed(1337)
    head = DNPHead(16, 2)
    names = list(head._modules.keys())
    assert names == [
        'projection_head', 'prediction_head', 'latent_path', 'deterministic_path',
        'mean_net', 'log_var_net', 'fc_decoder', 'classifier',
        'contrastive_class_selector_0', 'contrastive_class_selector_1',
        'contrastive_class_selector_memory0', 'contrastive_class_selector_memory1',
    ]


def test_best_score_is_rounded_like_caai_after_validation():
    source = Path('ssnet_final/engine/runner.py').read_text(encoding='utf-8')
    assert 'best_score = round(float(score), 4)' in source
    assert 'best_updated = (not best_initialized) or (score > best_score)' in source
