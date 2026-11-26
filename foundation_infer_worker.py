"""Isolated one-shot Transformers inference worker.

When GraphRAG is enabled this worker loads the VLM only once, then performs:
1) image-grounded clue extraction;
2) question + image clue fusion;
3) local Hybrid GraphRAG retrieval;
4) final answer generation with the original image + original question + evidence.

The worker exits afterwards so GPU memory is released, preserving the existing
on-demand inference behaviour of the project.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from foundation_model.inference.transformers_backend import LoadedTransformersVLM
from foundation_model.rag.index import GraphRAGIndexManager
from foundation_model.rag.local_backend import load_cached_knowledge_base
from foundation_model.rag.multimodal import (
    build_final_prompt,
    build_multimodal_query,
    build_visual_clue_prompt,
)
from foundation_model.rag.pipeline import HybridGraphRAGPipeline


def _combine_profiles(first: dict | None, second: dict | None) -> dict:
    first = first or {}
    second = second or {}
    total_latency = float(first.get("total_latency_s") or 0.0) + float(second.get("total_latency_s") or 0.0)
    prompt_tokens = int(first.get("prompt_tokens") or 0) + int(second.get("prompt_tokens") or 0)
    completion_tokens = int(first.get("completion_tokens") or 0) + int(second.get("completion_tokens") or 0)
    peak = max(float(first.get("peak_vram_mb") or 0.0), float(second.get("peak_vram_mb") or 0.0))
    return {
        "total_latency_s": total_latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_second": (completion_tokens / total_latency) if total_latency > 0 else 0.0,
        "peak_vram_mb": peak,
        "stage1_visual_latency_s": float(first.get("total_latency_s") or 0.0),
        "stage2_answer_latency_s": float(second.get("total_latency_s") or 0.0),
    }


def _retrieve(request: dict, question: str, visual_clues: str):
    index_dir = str(request.get("knowledge_base_dir") or "").strip()
    if not index_dir:
        raise RuntimeError("医学知识增强已启用，但没有设置 GraphRAG 索引目录。")
    manager = GraphRAGIndexManager(index_dir)
    if not manager.ready:
        raise RuntimeError("医学知识增强已启用，但知识库尚未构建。请先在④医学知识增强中构建知识库。")

    plan = build_multimodal_query(question, visual_clues)
    kb = load_cached_knowledge_base(str(Path(index_dir).expanduser().resolve()), manager.fingerprint())
    context = HybridGraphRAGPipeline(kb).retrieve(
        plan.retrieval_query,
        top_k=int(request.get("rag_top_k", 5)),
        graph_hops=int(request.get("rag_graph_hops", 1)),
        max_context_chars=int(request.get("rag_context_max_chars", 9000)),
    )
    return plan, context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))

    runtime = LoadedTransformersVLM(
        base_model=request["base_model"],
        adapter_path=request.get("adapter_path"),
        gpu_id=0,
        precision=request.get("precision", "bf16"),
        trust_remote_code=request.get("trust_remote_code", False),
        quantization=request.get("quantization", "4bit_nf4"),
    )

    try:
        rag_enabled = bool(request.get("rag_enabled", False))
        if not rag_enabled:
            text, profile = runtime.generate(
                image_path=request["image_path"],
                prompt=request["prompt"],
                max_new_tokens=request.get("max_new_tokens", 256),
                temperature=request.get("temperature", 0.0),
                history=request.get("history") or [],
            )
            payload = {"text": text, "profile": profile, "rag": None}
        else:
            # Stage 1: use the image. This is retrieval planning, not final diagnosis.
            visual_prompt = build_visual_clue_prompt(request["prompt"])
            visual_text, visual_profile = runtime.generate(
                image_path=request["image_path"],
                prompt=visual_prompt,
                max_new_tokens=int(request.get("rag_visual_max_new_tokens", 192)),
                temperature=0.0,
                history=None,
            )

            # Stage 2: fuse question + visual clues, then retrieve Hybrid GraphRAG evidence.
            plan, context = _retrieve(request, request["prompt"], visual_text)

            # Stage 3: show the original image again and answer with retrieved evidence.
            final_prompt = build_final_prompt(request["prompt"], context.evidence_text)
            text, final_profile = runtime.generate(
                image_path=request["image_path"],
                prompt=final_prompt,
                max_new_tokens=request.get("max_new_tokens", 256),
                temperature=request.get("temperature", 0.0),
                history=None,
            )
            payload = {
                "text": text,
                "profile": _combine_profiles(visual_profile, final_profile),
                "rag": {
                    "visual_clues": plan.visual_clues_text,
                    "question_keywords": list(plan.question_keywords),
                    "visual_keywords": list(plan.visual_keywords),
                    "retrieval_query": plan.retrieval_query,
                    "sources": list(context.sources),
                },
            }

        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
