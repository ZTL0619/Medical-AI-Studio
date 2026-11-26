import json
import math
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

from ssnet_final.engine.metrics import average_metrics, binary_metrics
from ssnet_final.data.ua_mt import case_h5_path, read_split


def evaluate(config, model, split="test", save_predictions=False, postprocess_lcc=False):
    if config.dataset != "LA":
        raise ValueError("Only the 3-D LA evaluation pipeline is enabled.")
    return evaluate_la(config, model, save_predictions, postprocess_lcc=postprocess_lcc)


def evaluate_la(config, model, save_predictions, postprocess_lcc=False):
    cases = read_split(config.data_root, "test")
    case_metrics = []
    prediction_dir = config.output / "predictions"
    if save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for case in tqdm(cases, desc="LA evaluation", leave=False):
        path = case_h5_path(config.data_root, case)
        with h5py.File(path, "r") as handle:
            image = handle["image"][:]
            label = handle["label"][:]
        prediction, score = infer_la_volume(config, model, image, 18, 4)
        # CCAI training validation (var_all_case_LA) uses the raw prediction,
        # while test_LA.py defaults to --nms 1.  Keep these two paths distinct.
        if postprocess_lcc:
            prediction = largest_connected_component(prediction)
        metrics = binary_metrics(prediction.astype(bool), label.astype(bool))
        case_metrics.append(metrics)
        if save_predictions:
            nib.save(
                nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                prediction_dir / f"{case}_pred.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(score.astype(np.float32), np.eye(4)),
                prediction_dir / f"{case}_scores.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(label.astype(np.float32), np.eye(4)),
                prediction_dir / f"{case}_gt.nii.gz",
            )
    result = average_metrics(case_metrics)
    return {"mean": result, "cases": dict(zip(cases, case_metrics))}


def _segmentation_probability(config, model_output):
    branches = model_output["branches"]
    if config.model == "catnet3d":
        positive = branches[0][:, :1].clamp(0.0, 1.0)
        return torch.cat([1.0 - positive, positive], dim=1)

    # The FSL reference validation uses decoder 2. DNP/default uses decoder 1.
    branch_index = 1 if config.feature_learning == "fsl" and config.semi_supervision != "dnp" else 0
    return F.softmax(branches[branch_index], dim=1)


def infer_la_volume(config, model, image, stride_xy, stride_z):
    device = config.device
    num_classes = config.num_classes
    patch_size = config.patch_size
    original_shape = image.shape
    pads = []
    for current, target in zip(image.shape, patch_size):
        total = max(target - current, 0)
        left = total // 2
        pads.append((left, total - left))
    if any(left or right for left, right in pads):
        image = np.pad(image, pads, mode="constant")
    width, height, depth = image.shape
    steps = [
        math.ceil((width - patch_size[0]) / stride_xy) + 1,
        math.ceil((height - patch_size[1]) / stride_xy) + 1,
        math.ceil((depth - patch_size[2]) / stride_z) + 1,
    ]
    score = np.zeros((num_classes,) + image.shape, dtype=np.float32)
    count = np.zeros(image.shape, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for xi in range(steps[0]):
            xs = min(stride_xy * xi, width - patch_size[0])
            for yi in range(steps[1]):
                ys = min(stride_xy * yi, height - patch_size[1])
                for zi in range(steps[2]):
                    zs = min(stride_z * zi, depth - patch_size[2])
                    patch = image[
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    tensor = torch.from_numpy(patch[None, None].astype(np.float32)).to(device)
                    model_output = model(tensor)
                    probability = _segmentation_probability(config, model_output)[0].cpu().numpy()
                    score[
                        :,
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ] += probability
                    count[
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ] += 1
    score = score / np.expand_dims(count, 0)
    prediction = np.argmax(score, axis=0)
    slices = tuple(slice(left, left + size) for (left, _), size in zip(pads, original_shape))
    return prediction[slices], score[(slice(None),) + slices]


def largest_connected_component(segmentation):
    """Match CCAI ``skimage.measure.label(segmentation)`` in 3-D.

    ``skimage.measure.label`` uses full connectivity by default; for a 3-D
    volume that is 26-neighbour connectivity.  ``scipy.ndimage.label`` without
    an explicit structure uses only 6-neighbour connectivity, which can split a
    diagonally connected LA mask into many pieces and select the wrong CC.
    """
    foreground = np.asarray(segmentation) > 0
    if not np.any(foreground):
        return np.asarray(segmentation)
    structure = ndimage.generate_binary_structure(foreground.ndim, foreground.ndim)
    labeled, count = ndimage.label(foreground, structure=structure)
    if count == 0:
        return np.asarray(segmentation)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return (labeled == largest).astype(np.asarray(segmentation).dtype)


def write_metrics(path, metrics):
    Path(path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
