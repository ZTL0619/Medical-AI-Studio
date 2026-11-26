"""Dependency-free baseline metrics for local smoke benchmarks."""
import re


def normalize_answer(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def exact_match(prediction, reference):
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction, reference):
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = 0
    remaining = list(ref)
    for token in pred:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)
