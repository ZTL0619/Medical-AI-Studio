"""Small local benchmark runner for the vLLM endpoint."""
import json
import time
from pathlib import Path

from foundation_model.data.jsonl_dataset import load_jsonl_samples
from foundation_model.evaluation.simple_metrics import exact_match, token_f1
from foundation_model.inference.vllm_client import chat_completion


def run_vllm_benchmark(dataset_path, endpoint, model, output_path, limit=20):
    samples = load_jsonl_samples(dataset_path)[: int(limit)]
    rows = []
    started = time.time()
    for sample in samples:
        result = chat_completion(endpoint, model, sample.question, image_path=sample.image, temperature=0.0)
        pred = result["text"]
        rows.append({
            "id": sample.sample_id,
            "image": str(sample.image),
            "question": sample.question,
            "reference": sample.answer,
            "prediction": pred,
            "exact_match": exact_match(pred, sample.answer),
            "token_f1": token_f1(pred, sample.answer),
            "profile": result["profile"],
        })
    summary = {
        "samples": len(rows),
        "exact_match": sum(r["exact_match"] for r in rows) / len(rows) if rows else 0.0,
        "token_f1": sum(r["token_f1"] for r in rows) / len(rows) if rows else 0.0,
        "avg_latency_s": sum(r["profile"]["total_latency_s"] for r in rows) / len(rows) if rows else 0.0,
        "avg_tokens_per_second": sum(r["profile"]["tokens_per_second"] for r in rows) / len(rows) if rows else 0.0,
        "elapsed_s": time.time() - started,
    }
    payload = {"summary": summary, "predictions": rows}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
