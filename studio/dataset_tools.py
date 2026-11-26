from pathlib import Path

import h5py
import numpy as np

from ssnet_final.data.ua_mt import (
    case_h5_path,
    layout_status,
    read_split,
    resolve_data_root,
)


def list_cases(data_root, split="test"):
    return read_split(data_root, split)


def case_path(data_root, case):
    return case_h5_path(data_root, case)


def load_case(data_root, case):
    path = case_path(data_root, case)
    with h5py.File(path, "r") as handle:
        image = handle["image"][:]
        label = handle["label"][:]
    return image, label


def inspect_dataset(data_root, sample_limit=3, labeled_percent=None):
    """Inspect a UA-MT style LA dataset.

    ``data_root`` should normally point at the repository's ``data`` directory.
    The function also accepts the ``2018LA_Seg_Training Set`` directory itself.
    """
    status = layout_status(data_root)
    root = resolve_data_root(data_root)
    issues = []

    if not status["root_exists"]:
        issues.append("数据目录不存在。")
    if not status["train_list_exists"]:
        issues.append(f"未找到 train.list: {status['train_list']}")
    if not status["test_list_exists"]:
        issues.append(f"未找到 test.list: {status['test_list']}")
    if not status["case_dir_exists"]:
        issues.append(
            "未找到 UA-MT 病例目录 '2018LA_Seg_Training Set'。"
        )

    train_cases = read_split(root, "train")
    test_cases = read_split(root, "test")

    missing_case_files = [
        (case, case_h5_path(root, case))
        for case in (train_cases + test_cases)
        if not case_h5_path(root, case).is_file()
    ]
    if missing_case_files:
        preview = ", ".join(case for case, _ in missing_case_files[:5])
        suffix = "..." if len(missing_case_files) > 5 else ""
        issues.append(
            f"train.list/test.list 中有 {len(missing_case_files)} 个病例缺少 mri_norm2.h5: {preview}{suffix}"
        )

    # UA-MT uses 80 training and 20 testing cases. Keep other counts readable,
    # but surface the mismatch so users know the split differs from the paper.
    if train_cases and len(train_cases) != 80:
        issues.append(f"train.list 包含 {len(train_cases)} 个病例；UA-MT 标准划分为 80 个。")
    if test_cases and len(test_cases) != 20:
        issues.append(f"test.list 包含 {len(test_cases)} 个病例；UA-MT 标准划分为 20 个。")

    samples = []
    for case in (train_cases + test_cases)[: max(1, int(sample_limit))]:
        path = case_h5_path(root, case)
        if not path.exists():
            issues.append(f"缺少病例文件: {path}")
            continue
        try:
            with h5py.File(path, "r") as handle:
                missing_keys = [key for key in ("image", "label") if key not in handle]
                if missing_keys:
                    issues.append(f"{case}: mri_norm2.h5 缺少数据集 {missing_keys}。")
                    continue
                image_shape = tuple(int(v) for v in handle["image"].shape)
                label_shape = tuple(int(v) for v in handle["label"].shape)
                labels = np.unique(handle["label"][:]).tolist()
            if image_shape != label_shape:
                issues.append(f"{case}: image/label 尺寸不一致。")
            samples.append(
                {
                    "case": case,
                    "file": str(path),
                    "image_shape": list(image_shape),
                    "label_shape": list(label_shape),
                    "label_values": [int(v) for v in labels],
                }
            )
        except Exception as exc:
            issues.append(f"{case}: 读取失败 ({type(exc).__name__}: {exc})")

    labeled_count = None
    unlabeled_count = None
    if labeled_percent is not None and train_cases:
        percent = int(labeled_percent)
        # Preserve the experiment definitions used by the integrated methods.
        standard_counts = {5: 4, 10: 8, 100: len(train_cases)}
        if percent in standard_counts:
            labeled_count = min(standard_counts[percent], len(train_cases))
            unlabeled_count = len(train_cases) - labeled_count

    return {
        "format": "UA-MT LA (mri_norm2.h5)",
        "data_root": str(root),
        "case_dir": str(status["case_dir"]),
        "train_cases": len(train_cases),
        "test_cases": len(test_cases),
        "labeled_percent": int(labeled_percent) if labeled_percent is not None else None,
        "labeled_cases": labeled_count,
        "unlabeled_cases": unlabeled_count,
        "samples": samples,
        "issues": issues,
        "valid": len(issues) == 0,
    }


def normalize_slice(array):
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(array[finite], [1, 99])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def overlay_slice(image_slice, mask_slice):
    base = normalize_slice(image_slice)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    mask = np.asarray(mask_slice) > 0
    if np.any(mask):
        tint = np.zeros_like(rgb)
        tint[..., 0] = 255
        rgb[mask] = 0.55 * rgb[mask] + 0.45 * tint[mask]
    return np.clip(rgb, 0, 255).astype(np.uint8)
