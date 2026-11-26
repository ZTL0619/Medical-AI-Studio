from torch.utils.data import DataLoader, DistributedSampler, Subset

from ssnet_final.data.common import (
    Compose,
    DistributedTwoStreamBatchSampler,
    TwoStreamBatchSampler,
    seed_worker,
)
from ssnet_final.data.la import LAHeartDataset, RandomCrop3D, RandomRotFlip3D, ToTensor3D


def build_train_loader(config, distributed=False, rank=0, world_size=1):
    if config.dataset != "LA":
        raise ValueError("Only the 3-D LA dataset is enabled in this project.")

    dataset = LAHeartDataset(
        config.data_root,
        "train",
        Compose([RandomRotFlip3D(), RandomCrop3D(config.patch_size), ToTensor3D()]),
    )

    if config.semi_supervision == "none":
        subset = Subset(dataset, list(range(config.labeled)))
        sampler = None
        if distributed:
            sampler = DistributedSampler(
                subset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=config.seed,
                drop_last=False,
            )
        return DataLoader(
            subset,
            batch_size=config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=config.workers,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=seed_worker(config.seed, rank),
        )

    labeled_indices = list(range(config.labeled))
    unlabeled_indices = list(range(config.labeled, len(dataset)))
    if distributed:
        sampler = DistributedTwoStreamBatchSampler(
            labeled_indices,
            unlabeled_indices,
            batch_size=config.batch_size,
            labeled_batch_size=config.labeled_batch_size,
            num_replicas=world_size,
            rank=rank,
            seed=config.seed,
        )
    else:
        sampler = TwoStreamBatchSampler(
            labeled_indices,
            unlabeled_indices,
            config.batch_size,
            config.batch_size - config.labeled_batch_size,
        )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker(config.seed, rank),
    )


def set_loader_epoch(loader, epoch):
    candidates = [getattr(loader, "sampler", None), getattr(loader, "batch_sampler", None)]
    for sampler in candidates:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
