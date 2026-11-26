"""End-user medical multimodal chat workspace.

When medical knowledge enhancement is enabled the request path is:
image + question -> visual clue extraction -> multimodal GraphRAG retrieval ->
original image + original question + retrieved evidence -> final answer.
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from foundation_model.config import FoundationServingConfig
from foundation_model.inference.vllm_client import chat_completion
from foundation_model.rag.index import GraphRAGIndexManager
from foundation_model.rag.local_backend import load_cached_knowledge_base
from foundation_model.rag.multimodal import (
    build_final_prompt,
    build_multimodal_query,
    build_visual_clue_prompt,
)
from foundation_model.rag.pipeline import HybridGraphRAGPipeline
from foundation_model.serving.vllm_manager import VLLMServiceManager
from studio.foundation_runtime import TransformersInferenceManager


def build_serving_config(draft: dict) -> FoundationServingConfig:
    adapter = Path(draft["adapter_path"]).expanduser() if draft.get("adapter_path") else None
    max_len = draft.get("max_model_len")
    return FoundationServingConfig(
        base_model=str(draft.get("base_model") or "").strip(),
        adapter_path=adapter,
        gpu_ids=tuple(int(i) for i in draft.get("gpu_ids", [])),
        strategy=str(draft.get("serving_strategy", "data_parallel")),
        precision=str(draft.get("serving_precision", "bf16")),
        host=str(draft.get("serving_host", "127.0.0.1")),
        port=int(draft.get("serving_port", 8010)),
        served_model_name=str(draft.get("served_model_name", "medical-vlm")),
        adapter_name=str(draft.get("adapter_name", "medical-adapter")),
        gpu_memory_utilization=float(draft.get("gpu_memory_utilization", 0.88)),
        max_model_len=int(max_len) if max_len else None,
        trust_remote_code=bool(draft.get("trust_remote_code", False)),
    ).validate()


def _save_uploaded_image(project_root, uploaded):
    suffix = Path(uploaded.name).suffix or ".png"
    root = Path(project_root) / "workspace" / "foundation" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{int(time.time() * 1000)}{suffix}"
    path.write_bytes(uploaded.getvalue())
    return path


def _conversation_for_model(messages):
    return [
        {"role": item["role"], "content": item["content"]}
        for item in messages
        if item.get("role") in {"user", "assistant", "system"} and item.get("content")
    ]


def _combine_profiles(first: dict | None, second: dict | None):
    first = first or {}
    second = second or {}
    latency = float(first.get("total_latency_s") or 0.0) + float(second.get("total_latency_s") or 0.0)
    completion = int(first.get("completion_tokens") or 0) + int(second.get("completion_tokens") or 0)
    return {
        "total_latency_s": latency,
        "prompt_tokens": int(first.get("prompt_tokens") or 0) + int(second.get("prompt_tokens") or 0),
        "completion_tokens": completion,
        "tokens_per_second": completion / latency if latency > 0 else 0.0,
        "peak_vram_mb": max(float(first.get("peak_vram_mb") or 0), float(second.get("peak_vram_mb") or 0)),
        "stage1_visual_latency_s": float(first.get("total_latency_s") or 0.0),
        "stage2_answer_latency_s": float(second.get("total_latency_s") or 0.0),
    }


def _render_profile(profile):
    if not profile:
        return
    latency = float(profile.get("total_latency_s") or 0.0)
    speed = float(profile.get("tokens_per_second") or 0.0)
    completion = int(profile.get("completion_tokens") or 0)
    parts = []
    if latency > 0:
        parts.append(f"{latency:.2f}s")
    if speed > 0:
        parts.append(f"{speed:.1f} tok/s")
    if completion > 0:
        parts.append(f"{completion} tokens")
    if parts:
        st.caption(" · ".join(parts))


def _render_rag_trace(rag):
    if not rag:
        return
    with st.expander("GraphRAG 检索过程与医学证据", expanded=False):
        visual = str(rag.get("visual_clues") or "").strip()
        if visual:
            st.markdown("**图像检索线索（仅用于检索，不等于最终诊断）**")
            st.code(visual, language="json")
        query = str(rag.get("retrieval_query") or "").strip()
        if query:
            st.markdown("**融合检索 Query（问题 + 图像）**")
            st.code(query, language="text")
        sources = rag.get("sources") or []
        if sources:
            st.markdown("**医学知识证据**")
        for source in sources:
            meta = source.get("metadata") or {}
            location = f" · p.{meta['page']}" if meta.get("page") else ""
            st.markdown(
                f"**[{source.get('index')}] {source.get('source', '')}{location}**  "
                f"score={float(source.get('score') or 0.0):.3f} · {source.get('retrieval_path', 'hybrid')}"
            )
            entities = source.get("graph_entities") or []
            if entities:
                st.caption("图路径概念：" + "、".join(str(x) for x in entities[:12]))
            text = str(source.get("text") or "").strip()
            if text:
                st.caption(text[:1000])


def _retrieve_multimodal_context(draft, question, visual_clues):
    index_dir = str(draft.get("knowledge_base_dir") or "").strip()
    if not index_dir:
        raise RuntimeError("医学知识增强已启用，但没有设置 GraphRAG 索引目录。")
    manager = GraphRAGIndexManager(index_dir)
    if not manager.ready:
        raise RuntimeError("医学知识增强已启用，但知识库尚未构建。请在左侧④先构建知识库。")
    plan = build_multimodal_query(question, visual_clues)
    kb = load_cached_knowledge_base(str(Path(index_dir).expanduser().resolve()), manager.fingerprint())
    context = HybridGraphRAGPipeline(kb).retrieve(
        plan.retrieval_query,
        top_k=int(draft.get("rag_top_k", 5)),
        graph_hops=int(draft.get("rag_graph_hops", 1)),
        max_context_chars=int(draft.get("rag_context_max_chars", 9000)),
    )
    return plan, context


def render_foundation_workspace(draft, project_root):
    service_manager = VLLMServiceManager(project_root)
    transformers_manager = TransformersInferenceManager(project_root)

    st.subheader("医疗多模态助手")
    backend = str(draft.get("inference_backend", "vllm"))
    gpu_ids = [int(i) for i in draft.get("gpu_ids", [])]
    service = service_manager.status()

    if backend == "vllm":
        if service.get("healthy"):
            st.caption("● 模型已就绪")
            ready = True
        elif service.get("state") == "running":
            st.caption("● 模型正在加载，请稍后再试")
            ready = False
        else:
            st.caption("○ 模型未加载，请在左侧“推理”中点击“加载模型”")
            ready = False
    else:
        ready = bool(str(draft.get("base_model") or "").strip() and gpu_ids)
        if ready:
            q = "4-bit NF4" if draft.get("inference_quantization", "4bit_nf4") == "4bit_nf4" else "未量化"
            st.caption(f"● 兼容模式：提交问题时自动加载模型 · {q}")
        elif not str(draft.get("base_model") or "").strip():
            st.caption("○ 请先选择模型 / 本地路径")
        else:
            st.caption("○ 请至少选择一张 GPU")

    uploaded = st.file_uploader(
        "添加医学图像",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        key="fm_chat_image",
        help="医学图像 + 用户问题；启用④后，图像线索和问题关键词会共同参与 GraphRAG 检索。",
    )
    if uploaded is not None:
        image_col, info_col = st.columns([1, 3], vertical_alignment="center")
        with image_col:
            st.image(uploaded, width=220)
        with info_col:
            st.markdown(f"**当前图像：** `{uploaded.name}`")
            st.caption("后续问题会继续使用这张图像，直到你替换或清空它。")

    if "fm_chat_messages" not in st.session_state:
        st.session_state.fm_chat_messages = []

    for item in st.session_state.fm_chat_messages:
        with st.chat_message(item.get("role", "assistant")):
            st.markdown(item.get("content", ""))
            _render_profile(item.get("profile"))
            _render_rag_trace(item.get("rag"))

    disabled_reason = None
    if uploaded is None:
        disabled_reason = "请先添加医学图像。"
    elif not ready:
        disabled_reason = "当前推理模型尚未就绪。"

    prompt = st.chat_input(
        "输入关于医学图像的问题……",
        disabled=bool(disabled_reason),
        key="fm_chat_prompt",
    )
    if disabled_reason:
        st.caption(disabled_reason)

    if prompt:
        previous = list(st.session_state.fm_chat_messages)
        image_path = _save_uploaded_image(project_root, uploaded)
        st.session_state.fm_chat_messages.append({"role": "user", "content": str(prompt)})
        try:
            rag_payload = None
            with st.spinner("正在分析医学图像并检索医学知识……" if draft.get("rag_enabled") else "正在分析医学图像……"):
                if backend == "vllm":
                    service = service_manager.status()
                    if not service.get("healthy"):
                        raise RuntimeError("模型服务尚未就绪，请在左侧“推理”中加载模型。")

                    if bool(draft.get("rag_enabled", False)):
                        visual_result = chat_completion(
                            service["endpoint"],
                            service["model_for_request"],
                            build_visual_clue_prompt(prompt),
                            image_path=image_path,
                            max_tokens=int(draft.get("rag_visual_max_new_tokens", 192)),
                            temperature=0.0,
                            history=None,
                        )
                        plan, context = _retrieve_multimodal_context(draft, prompt, visual_result.get("text", ""))
                        result = chat_completion(
                            service["endpoint"],
                            service["model_for_request"],
                            build_final_prompt(prompt, context.evidence_text),
                            image_path=image_path,
                            max_tokens=int(draft.get("max_new_tokens", 256)),
                            temperature=float(draft.get("temperature", 0.0)),
                            history=None,
                        )
                        result["profile"] = _combine_profiles(visual_result.get("profile"), result.get("profile"))
                        rag_payload = {
                            "visual_clues": plan.visual_clues_text,
                            "question_keywords": list(plan.question_keywords),
                            "visual_keywords": list(plan.visual_keywords),
                            "retrieval_query": plan.retrieval_query,
                            "sources": list(context.sources),
                        }
                    else:
                        result = chat_completion(
                            service["endpoint"],
                            service["model_for_request"],
                            prompt,
                            image_path=image_path,
                            max_tokens=int(draft.get("max_new_tokens", 256)),
                            temperature=float(draft.get("temperature", 0.0)),
                            history=_conversation_for_model(previous),
                        )
                else:
                    if not gpu_ids:
                        raise RuntimeError("请至少选择一张 GPU。")
                    result = transformers_manager.run(
                        base_model=str(draft.get("base_model") or "").strip(),
                        image_path=image_path,
                        prompt=prompt,
                        adapter_path=draft.get("adapter_path") or None,
                        gpu_id=gpu_ids[0],
                        precision=str(draft.get("inference_precision", "bf16")),
                        quantization=str(draft.get("inference_quantization", "4bit_nf4")),
                        max_new_tokens=int(draft.get("max_new_tokens", 256)),
                        temperature=float(draft.get("temperature", 0.0)),
                        trust_remote_code=bool(draft.get("trust_remote_code", False)),
                        history=_conversation_for_model(previous),
                        rag_enabled=bool(draft.get("rag_enabled", False)),
                        knowledge_base_dir=draft.get("knowledge_base_dir") or None,
                        rag_top_k=int(draft.get("rag_top_k", 5)),
                        rag_graph_hops=int(draft.get("rag_graph_hops", 1)),
                        rag_context_max_chars=int(draft.get("rag_context_max_chars", 9000)),
                        rag_visual_max_new_tokens=int(draft.get("rag_visual_max_new_tokens", 192)),
                    )
                    rag_payload = result.get("rag")

            st.session_state.fm_chat_messages.append({
                "role": "assistant",
                "content": str(result.get("text", "")),
                "profile": result.get("profile") or {},
                "rag": rag_payload,
            })
        except Exception as exc:
            st.session_state.fm_chat_messages.append({
                "role": "assistant",
                "content": f"推理失败：{exc}",
                "profile": {},
                "rag": None,
            })
        st.rerun()
