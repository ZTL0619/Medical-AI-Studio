"""Interfaces for a future lightweight medical Hybrid GraphRAG knowledge base.

The framework is intentionally implemented now while ingestion/index construction
is deferred. This keeps model/RAG coupling stable when a real knowledge base is
added later.
"""
from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEntity:
    entity_id: str
    name: str
    entity_type: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraphRelation:
    source_id: str
    relation: str
    target_id: str
    evidence_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalHit:
    chunk: DocumentChunk
    score: float
    retrieval_path: str = "vector"
    graph_entities: tuple[str, ...] = ()


class VectorStore(Protocol):
    def search(self, query: str, top_k: int) -> Sequence[RetrievalHit]: ...


class GraphStore(Protocol):
    def find_entities(self, query: str) -> Sequence[GraphEntity]: ...
    def expand(self, entity_ids: Sequence[str], hops: int) -> Sequence[GraphRelation]: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[RetrievalHit], top_k: int) -> Sequence[RetrievalHit]: ...


class MedicalKnowledgeBase(Protocol):
    @property
    def ready(self) -> bool: ...
    def retrieve(self, query: str, top_k: int = 5, graph_hops: int = 1) -> Sequence[RetrievalHit]: ...
