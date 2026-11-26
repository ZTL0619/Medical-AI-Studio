import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ssnet_final.data.ua_mt import case_h5_path, read_split, resolve_data_root


class LAHeartDataset(Dataset):
    """3-D LA dataset reader compatible with the UA-MT ``data`` directory."""

    def __init__(self, root, split, transform=None):
        self.root = resolve_data_root(root)
        self.transform = transform
        self.cases = read_split(self.root, split)
        if not self.cases:
            raise FileNotFoundError(
                f"未找到有效的 {split}.list: {self.root / (split + '.list')}"
            )

    def __len__(self):
        return len(self.cases)

    def case_path(self, index):
        return case_h5_path(self.root, self.cases[index])

    def __getitem__(self, index):
        path = self.case_path(index)
        with h5py.File(path, "r") as handle:
            image = handle["image"][:]
            label = handle["label"][:]
        sample = {"image": image, "label": label, "case": self.cases[index]}
        return self.transform(sample) if self.transform is not None else sample


class RandomRotFlip3D:
    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return {**sample, "image": image, "label": label}


class RandomCrop3D:
    def __init__(self, output_size):
        self.output_size = tuple(output_size)

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]
        pads = []
        for current, target in zip(label.shape, self.output_size):
            amount = max((target - current) // 2 + 3, 0) if current <= target else 0
            pads.append((amount, amount))
        if any(left or right for left, right in pads):
            image = np.pad(image, pads, mode="constant")
            label = np.pad(label, pads, mode="constant")
        starts = [np.random.randint(0, current - target) for current, target in zip(image.shape, self.output_size)]
        slices = tuple(slice(start, start + target) for start, target in zip(starts, self.output_size))
        return {**sample, "image": image[slices], "label": label[slices]}


class ToTensor3D:
    def __call__(self, sample):
        image = torch.from_numpy(sample["image"].reshape((1,) + sample["image"].shape).astype(np.float32))
        label = torch.from_numpy(sample["label"].astype(np.int64))
        return {**sample, "image": image, "label": label}
