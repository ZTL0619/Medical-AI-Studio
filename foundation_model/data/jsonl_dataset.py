"""Load a small local medical image+text SFT dataset.

Expected JSONL fields per line:
  image, question, answer[, system, id]
Image paths may be absolute or relative to the JSONL file.
"""
import json
from pathlib import Path

from foundation_model.data.schema import VQASample


def load_jsonl_samples(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
        for key in ("image", "question", "answer"):
            if not str(item.get(key, "")).strip():
                raise ValueError(f"Missing '{key}' at {path}:{line_no}")
        image = Path(item["image"]).expanduser()
        if not image.is_absolute():
            image = (path.parent / image).resolve()
        rows.append(VQASample(
            image=image,
            question=str(item["question"]),
            answer=str(item["answer"]),
            system=str(item.get("system") or "You are a medical visual assistant. Answer carefully and concisely."),
            sample_id=str(item.get("id")) if item.get("id") is not None else None,
        ))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def validate_jsonl_dataset(path, check_images=True):
    samples = load_jsonl_samples(path)
    missing = [str(s.image) for s in samples if check_images and not s.image.exists()]
    return {
        "samples": len(samples),
        "missing_images": missing,
        "valid": not missing,
        "first_sample": {
            "id": samples[0].sample_id,
            "image": str(samples[0].image),
            "question": samples[0].question,
            "answer": samples[0].answer,
        },
    }
