import numpy as np
from medpy import metric


def binary_metrics(prediction, target):
    if np.sum(prediction) == 0 or np.sum(target) == 0:
        return {"dice": 0.0, "jaccard": 0.0, "asd": 0.0, "hd95": 0.0}
    return {
        "dice": float(metric.binary.dc(prediction, target)),
        "jaccard": float(metric.binary.jc(prediction, target)),
        "asd": float(metric.binary.asd(prediction, target)),
        "hd95": float(metric.binary.hd95(prediction, target)),
    }


def average_metrics(items):
    if not items:
        return {"dice": 0.0, "jaccard": 0.0, "asd": 0.0, "hd95": 0.0}
    return {key: float(np.mean([item[key] for item in items])) for key in items[0]}
