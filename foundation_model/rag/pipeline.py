"""RAG orchestration independent of the concrete vector/graph database."""
from dataclasses import dataclass

from foundation_model.rag.null_backend import NullMedicalKnowledgeBase


@dataclass
class RAGContext:
    query: str
    evidence_text: str
    sources: list[dict]


class HybridGraphRAGPipeline:
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base or NullMedicalKnowledgeBase()

    @property
    def ready(self):
        return bool(self.knowledge_base.ready)

    def retrieve(self, query, top_k=5, graph_hops=1, max_context_chars=9000):
        hits = list(self.knowledge_base.retrieve(query, top_k=top_k, graph_hops=graph_hops))
        evidence = []
        sources = []
        used = 0
        for index, hit in enumerate(hits, 1):
            text = str(hit.chunk.text or "").strip()
            if not text:
                continue
            remaining = max(0, int(max_context_chars) - used)
            if remaining <= 0:
                break
            clipped = text[:remaining]
            evidence.append(f"[{index}] {clipped}")
            used += len(clipped)
            sources.append({
                "index": index,
                "source": hit.chunk.source,
                "chunk_id": hit.chunk.chunk_id,
                "score": float(hit.score),
                "retrieval_path": hit.retrieval_path,
                "graph_entities": list(hit.graph_entities),
                "metadata": dict(hit.chunk.metadata or {}),
                "text": clipped,
            })
        return RAGContext(query=query, evidence_text="\n\n".join(evidence), sources=sources)

    @staticmethod
    def augment_prompt(question, context: RAGContext):
        if not context.evidence_text:
            return question
        return (
            f"{question}\n\n"
            "下面是从本地医学知识库检索到的证据。请把图像本身作为主要视觉依据，"
            "仅在证据与当前问题相关时使用它；不要把检索文本中未被图像或问题支持的内容强行套用。"
            "若证据不足或相互矛盾，请明确说明。使用某条证据时请用 [1]、[2] 这样的编号引用。\n\n"
            f"医学知识证据：\n{context.evidence_text}"
        )
