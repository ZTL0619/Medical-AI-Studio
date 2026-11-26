from pathlib import Path

from ssnet_final.config import build_config, config_from_json_dict
from ssnet_final.data.common import DistributedTwoStreamBatchSampler


def test_resolved_config_roundtrip_does_not_double_scale_lr():
    config = build_config(
        command="train",
        dataset="LA",
        model="vnet",
        feature_learning="fsl",
        semi_supervision="dnp",
        labeled_percent=10,
        data_root=Path("/tmp/la"),
        output=Path("/tmp/out"),
        parallel_mode="ddp",
        gpu_ids=(0, 1, 2, 3),
        linear_lr_scaling=True,
        batch_size=4,
        labeled_batch_size=2,
    )
    restored = config_from_json_dict(config.to_json_dict())
    assert restored.learning_rate == config.learning_rate
    assert restored.gpu_ids == config.gpu_ids
    assert restored.global_batch_size == 16


def test_distributed_two_stream_sampler_preserves_local_composition():
    labeled = list(range(8))
    unlabeled = list(range(8, 80))
    for rank in range(4):
        sampler = DistributedTwoStreamBatchSampler(
            labeled,
            unlabeled,
            batch_size=4,
            labeled_batch_size=2,
            num_replicas=4,
            rank=rank,
            seed=11,
        )
        sampler.set_epoch(2)
        batch = next(iter(sampler))
        assert len(batch) == 4
        assert all(index < 8 for index in batch[:2])
        assert all(index >= 8 for index in batch[2:])
