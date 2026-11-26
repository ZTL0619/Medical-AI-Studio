from pathlib import Path

from foundation_model.rag.multimodal import build_multimodal_query, build_final_prompt


def test_multimodal_query_fuses_question_and_visual_clues():
    visual = '{"modality":"胸部X光","anatomy":["右下肺"],"findings":["片状高密度影"],"candidate_concepts":["肺实变","肺炎"]}'
    plan = build_multimodal_query("请帮我分析一下这个影像", visual)
    assert "请帮我分析一下这个影像" in plan.retrieval_query
    assert any(term in plan.retrieval_query for term in ["右下肺", "肺实变", "肺炎", "高密度"])
    assert plan.visual_keywords


def test_final_prompt_uses_evidence_but_not_first_stage_clues():
    prompt = build_final_prompt("分析这个影像", "[1] 肺实变常见于……")
    assert "分析这个影像" in prompt
    assert "[1]" in prompt
    assert "原始医学图像" in prompt


def test_transformers_worker_reuses_one_loaded_model_and_supports_nf4():
    worker = Path("foundation_infer_worker.py").read_text(encoding="utf-8")
    backend = Path("foundation_model/inference/transformers_backend.py").read_text(encoding="utf-8")
    runtime = Path("studio/foundation_runtime.py").read_text(encoding="utf-8")
    sidebar = Path("studio/ui/foundation_sidebar.py").read_text(encoding="utf-8")
    assert worker.count("LoadedTransformersVLM(") == 1
    assert "build_visual_clue_prompt" in worker
    assert "build_multimodal_query" in worker
    assert "build_final_prompt" in worker
    assert 'bnb_4bit_quant_type="nf4"' in backend
    assert "bnb_4bit_use_double_quant=True" in backend
    assert 'quantization="4bit_nf4"' in runtime
    assert "VLM 推理量化" in sidebar
