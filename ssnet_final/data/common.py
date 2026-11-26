import itertools
import random

import numpy as np
from torch.utils.data import Sampler


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class TwoStreamBatchSampler(Sampler):
    def __init__(self, labeled_indices, unlabeled_indices, batch_size, unlabeled_batch_size):
        self.labeled_indices = list(labeled_indices)
        self.unlabeled_indices = list(unlabeled_indices)
        self.unlabeled_batch_size = unlabeled_batch_size
        self.labeled_batch_size = batch_size - unlabeled_batch_size
        if not len(self.labeled_indices) >= self.labeled_batch_size > 0:
            raise ValueError("Invalid labeled batch configuration")
        if not len(self.unlabeled_indices) >= self.unlabeled_batch_size > 0:
            raise ValueError("Invalid unlabeled batch configuration")

    def __iter__(self):
        labeled = np.random.permutation(self.labeled_indices)
        unlabeled = itertools.chain.from_iterable(self._infinite_unlabeled())
        return (
            left + right
            for left, right in zip(
                self._groups(labeled, self.labeled_batch_size),
                self._groups(unlabeled, self.unlabeled_batch_size),
            )
        )

    def __len__(self):
        return len(self.labeled_indices) // self.labeled_batch_size

    def _infinite_unlabeled(self):
        while True:
            yield np.random.permutation(self.unlabeled_indices)

    @staticmethod
    def _groups(iterable, size):
        args = [iter(iterable)] * size
        return zip(*args)


class DistributedTwoStreamBatchSampler(Sampler):
    """DDP-aware labeled/unlabeled batch sampler.

    Every rank receives the same *composition* per local batch: the first
    ``labeled_batch_size`` samples are labeled and the remainder are unlabeled.
    A deterministic global batch is generated and then split by rank. Sampling
    with replacement is used when the small labeled pool is smaller than the
    global labeled batch, which keeps 5% LA training valid on more than two GPUs.
    """

    def __init__(
        self,
        labeled_indices,
        unlabeled_indices,
        batch_size,
        labeled_batch_size,
        num_replicas,
        rank,
        seed=0,
    ):
        self.labeled_indices = np.asarray(list(labeled_indices), dtype=np.int64)
        self.unlabeled_indices = np.asarray(list(unlabeled_indices), dtype=np.int64)
        self.batch_size = int(batch_size)
        self.labeled_batch_size = int(labeled_batch_size)
        self.unlabeled_batch_size = self.batch_size - self.labeled_batch_size
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 1:
            raise ValueError("Semi-supervised DDP requires local batch_size >= 2.")
        if not 0 < self.labeled_batch_size < self.batch_size:
            raise ValueError("Each rank must receive both labeled and unlabeled samples.")
        if len(self.labeled_indices) == 0 or len(self.unlabeled_indices) == 0:
            raise ValueError("Both labeled and unlabeled index pools are required.")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas).")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        global_labeled = self.labeled_batch_size * self.num_replicas
        return max(1, int(np.ceil(len(self.labeled_indices) / global_labeled)))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        global_labeled = self.labeled_batch_size * self.num_replicas
        global_unlabeled = self.unlabeled_batch_size * self.num_replicas

        for _ in range(len(self)):
            labeled = rng.choice(
                self.labeled_indices,
                size=global_labeled,
                replace=global_labeled > len(self.labeled_indices),
            )
            unlabeled = rng.choice(
                self.unlabeled_indices,
                size=global_unlabeled,
                replace=global_unlabeled > len(self.unlabeled_indices),
            )
            l_start = self.rank * self.labeled_batch_size
            u_start = self.rank * self.unlabeled_batch_size
            local = np.concatenate(
                [
                    labeled[l_start : l_start + self.labeled_batch_size],
                    unlabeled[u_start : u_start + self.unlabeled_batch_size],
                ]
            )
            yield local.tolist()


def seed_worker(seed, rank=0):
    """Match CCAI aux4 worker_init_fn exactly.

    The reference code only calls ``random.seed(args.seed + worker_id)``.
    It does not reseed NumPy in the DataLoader worker initializer.
    ``rank`` is retained only for API compatibility with the studio loader.
    """
    del rank

    def init(worker_id):
        random.seed(int(seed) + int(worker_id))

    return init
