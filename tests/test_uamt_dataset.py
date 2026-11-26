from pathlib import Path

import h5py
import numpy as np

from ssnet_final.data.la import LAHeartDataset
from ssnet_final.data.ua_mt import case_h5_path, resolve_data_root
from studio.dataset_tools import inspect_dataset, list_cases, load_case


def _make_uamt(root: Path):
    case_dir = root / "2018LA_Seg_Training Set"
    train = [f"train_{i:03d}" for i in range(80)]
    test = [f"test_{i:03d}" for i in range(20)]
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.list").write_text("\n".join(train) + "\n")
    (root / "test.list").write_text("\n".join(test) + "\n")
    for case in (train + test):
        folder = case_dir / case
        folder.mkdir(parents=True, exist_ok=True)
        with h5py.File(folder / "mri_norm2.h5", "w") as f:
            f.create_dataset("image", data=np.zeros((16, 16, 8), dtype=np.float32))
            f.create_dataset("label", data=np.zeros((16, 16, 8), dtype=np.uint8))
    return train, test


def test_uamt_layout_and_split(tmp_path):
    train, test = _make_uamt(tmp_path)
    assert list_cases(tmp_path, "train") == train
    assert list_cases(tmp_path, "test") == test
    assert case_h5_path(tmp_path, train[0]).name == "mri_norm2.h5"

    report = inspect_dataset(tmp_path, sample_limit=3, labeled_percent=10)
    assert report["valid"] is True
    assert report["train_cases"] == 80
    assert report["test_cases"] == 20
    assert report["labeled_cases"] == 8
    assert report["unlabeled_cases"] == 72

    image, label = load_case(tmp_path, train[0])
    assert image.shape == label.shape == (16, 16, 8)


def test_training_set_directory_can_be_selected(tmp_path):
    train, _ = _make_uamt(tmp_path)
    direct = tmp_path / "2018LA_Seg_Training Set"
    assert resolve_data_root(direct) == tmp_path
    dataset = LAHeartDataset(direct, "train")
    assert len(dataset) == 80
    assert dataset.cases[0] == train[0]
