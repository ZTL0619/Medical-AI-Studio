"""UA-MT Left Atrium dataset layout helpers.

The supported on-disk layout matches yulequan/UA-MT::data:

    data/
    ├── train.list
    ├── test.list
    └── 2018LA_Seg_Training Set/
        └── <case_id>/mri_norm2.h5

For convenience, callers may also pass the ``2018LA_Seg_Training Set``
directory itself; in that case its parent is treated as the data root.
"""

from pathlib import Path


TRAINING_DIR_NAMES = (
    "2018LA_Seg_Training Set",  # exact UA-MT repository folder name
    "2018LA_Seg_TrainingSet",   # tolerate the spelling used in some README copies
)


def resolve_data_root(path):
    """Return the UA-MT ``data`` directory for *path* when it can be inferred."""
    path = Path(path).expanduser()
    if path.name in TRAINING_DIR_NAMES and (path.parent / "train.list").exists():
        return path.parent
    return path


def training_data_dir(data_root):
    """Locate the directory containing case folders."""
    root = resolve_data_root(data_root)
    for name in TRAINING_DIR_NAMES:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    # Return the canonical path so downstream error messages are concrete.
    return root / TRAINING_DIR_NAMES[0]


def split_list_path(data_root, split):
    root = resolve_data_root(data_root)
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    return root / f"{split}.list"


def read_split(data_root, split):
    path = split_list_path(data_root, split)
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def case_h5_path(data_root, case):
    return training_data_dir(data_root) / str(case) / "mri_norm2.h5"


def layout_status(data_root):
    """Return structural information without opening every HDF5 file."""
    root = resolve_data_root(data_root)
    train_list = root / "train.list"
    test_list = root / "test.list"
    case_dir = training_data_dir(root)
    return {
        "root": root,
        "train_list": train_list,
        "test_list": test_list,
        "case_dir": case_dir,
        "root_exists": root.is_dir(),
        "train_list_exists": train_list.is_file(),
        "test_list_exists": test_list.is_file(),
        "case_dir_exists": case_dir.is_dir(),
    }
