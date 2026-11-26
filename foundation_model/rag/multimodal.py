"""Multimodal query planning for medical GraphRAG.

The important design rule is that retrieval is *not* driven by the user's text
alone.  The user question contributes intent/keywords, while a first VLM pass
extracts image-grounded medical clues.  These are fused into one retrieval
query, then the original image + original question + retrieved evidence are
sent to the VLM for the final answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from foundation_model.rag.index import extract_concepts


_JSON_RE = re.compile(r"\{.*\}", re.S)


@dataclass(frozen=True)
class MultimodalQueryPlan:
    question: str
    question_keywords: tuple[str, ...]
    visual_clues_text: str
    visual_keywords: tuple[str, ...]
    retrieval_query: str

    def to_dict(self):
        return asdict(self)


def build_visual_clue_prompt(question: str) -> str:
    """Prompt the VLM to describe only retrieval-useful image clues.

    The output is explicitly *not* treated as the final diagnosis.  It is used
    only to improve retrieval recall/precision for the subsequent GraphRAG
    search.
    """
    return f"""你现在只做医学图像检索线索提取，不要回答用户问题，也不要给出最终诊断。
请仅根据当前医学图像，提取适合用于医学知识库检索的客观或候选线索。

用户原始问题：{question}

请输出一个 JSON 对象，字段固定为：
{{
  "modality": "检查/成像模态；不知道则空字符串",
  "anatomy": ["器官或组织"],
  "location": ["解剖部位/左右侧/区域"],
  "findings": ["可见异常或形态特征"],
  "morphology": ["形状、密度、信号、染色、细胞/组织结构等"],
  "candidate_concepts": ["仅用于检索扩展的可能相关医学概念，不作为确诊"],
  "uncertainty": "对图像线索不确定性的简短说明"
}}

要求：
1. 只输出 JSON，不要解释。
2. 只写图像支持的内容；不确定就留空或明确不确定。
3. candidate_concepts 只是检索候选词，不是最终诊断。
"""


def _flatten_json_strings(value) -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            out.extend(_flatten_json_strings(item))
    elif isinstance(value, list):
        for item in value:
            out.extend(_flatten_json_strings(item))
    elif value is not None:
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def parse_visual_clues(raw_text: str) -> tuple[str, list[str]]:
    """Return normalized clue text and compact concept keywords.

    This is intentionally tolerant because different VLMs may wrap JSON in
    markdown fences or return almost-JSON.  If JSON parsing fails, the raw text
    still contributes concepts instead of breaking the whole inference path.
    """
    raw = str(raw_text or "").strip()
    if not raw:
        return "", []

    normalized = raw
    strings: list[str] = []
    match = _JSON_RE.search(raw)
    if match:
        candidate = match.group(0)
        try:
            payload = json.loads(candidate)
            strings = _flatten_json_strings(payload)
            normalized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            strings = []

    source_for_keywords = " ".join(strings) if strings else raw
    keywords = extract_concepts(source_for_keywords, max_terms=28)
    return normalized, keywords


def build_multimodal_query(question: str, visual_clues_text: str) -> MultimodalQueryPlan:
    """Fuse textual intent and image clues into one GraphRAG retrieval query."""
    question = str(question or "").strip()
    question_keywords = extract_concepts(question, max_terms=16)
    normalized_visual, visual_keywords = parse_visual_clues(visual_clues_text)

    # Preserve stable order and avoid overweighting repeated terms.
    merged: list[str] = []
    for item in list(question_keywords) + list(visual_keywords):
        item = str(item).strip()
        if item and item not in merged:
            merged.append(item)

    # Keep the original question because embeddings benefit from sentence-level
    # intent; add compact concepts afterwards to inject image-grounded context.
    pieces = []
    if question:
        pieces.append(question)
    if merged:
        pieces.append("医学检索关键词：" + "；".join(merged))

    retrieval_query = "\n".join(pieces).strip()
    return MultimodalQueryPlan(
        question=question,
        question_keywords=tuple(question_keywords),
        visual_clues_text=normalized_visual,
        visual_keywords=tuple(visual_keywords),
        retrieval_query=retrieval_query,
    )


def build_final_prompt(question: str, evidence_text: str) -> str:
    """Build the second-stage answer prompt.

    The first-stage visual clues are intentionally omitted here.  The VLM sees
    the original image again, so retrieval-planning hypotheses are not promoted
    to ground truth.  Only retrieved evidence is added to the user's question.
    """
    question = str(question or "").strip()
    evidence = str(evidence_text or "").strip()
    if not evidence:
        return question
    return f"""用户原始问题：
{question}

下面是根据“用户问题 + 当前图像视觉线索”从医学 GraphRAG 知识库检索到的证据：
{evidence}

请重新查看原始医学图像，并结合用户原始问题与上述证据作答。
要求：
1. 原始图像是当前病例的主要视觉依据，检索资料只是知识补充。
2. 不要把检索资料中的疾病强行套到图像上。
3. 证据不足、图像不典型或存在冲突时，要明确说明不确定性。
4. 使用检索证据时，用 [1]、[2] 这样的编号引用对应证据。
5. 不要把第一次图像线索提取当作已经确认的诊断。
"""
