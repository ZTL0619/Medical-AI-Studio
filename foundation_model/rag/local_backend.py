"""Concrete local Hybrid GraphRAG retrieval backend."""
from __future__ import annotations

from collections import defaultdict, deque
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from foundation_model.rag.contracts import DocumentChunk, RetrievalHit
from foundation_model.rag.embeddings import create_encoder
from foundation_model.rag.index import GraphRAGIndexManager, extract_concepts


class LocalHybridMedicalKnowledgeBase:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir).expanduser().resolve()
        manager = GraphRAGIndexManager(self.index_dir)
        if not manager.ready:
            raise RuntimeError(f"GraphRAG 知识库尚未构建：{self.index_dir}")
        self.manifest = json.loads((self.index_dir / manager.MANIFEST).read_text(encoding="utf-8"))
        self.chunks = [
            json.loads(line)
            for line in (self.index_dir / manager.CHUNKS).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.graph = json.loads((self.index_dir / manager.GRAPH).read_text(encoding="utf-8"))
        self.embeddings = np.load(self.index_dir / manager.EMBEDDINGS, mmap_mode="r")
        embedding_backend = self.manifest.get("embedding_backend", "hash")
        embedding_model = self.manifest.get("embedding_model")
        resolved_path = self.manifest.get("embedding_resolved_path")
        if resolved_path and Path(str(resolved_path)).expanduser().is_dir():
            embedding_backend = "local"
            embedding_model = str(Path(str(resolved_path)).expanduser().resolve())
        self.encoder = create_encoder(
            embedding_backend,
            embedding_model,
            dimension=int(self.manifest.get("embedding_dimension") or 768),
            device="cpu",
        )
        if self.embeddings.shape[0] != len(self.chunks):
            raise RuntimeError("GraphRAG 索引损坏：chunks 与 embeddings 数量不一致。")

    @property
    def ready(self):
        return True

    def _graph_scores(self, query: str, vector_order: list[int], hops: int):
        entity_to_chunks = self.graph.get("entity_to_chunks", {})
        neighbors = self.graph.get("neighbors", {})
        query_entities = [e for e in extract_concepts(query, max_terms=20) if e in entity_to_chunks]

        # Seed from the strongest vector chunks too. This makes graph expansion
        # useful when the query uses a synonym not present verbatim in the graph.
        seeded_from_vector: list[str] = []
        for idx in vector_order[:3]:
            for entity in self.chunks[idx].get("entities", [])[:8]:
                if entity in entity_to_chunks and entity not in seeded_from_vector:
                    seeded_from_vector.append(entity)
        seeds = list(dict.fromkeys(query_entities + seeded_from_vector[:10]))

        chunk_scores: defaultdict[int, float] = defaultdict(float)
        matched: defaultdict[int, set[str]] = defaultdict(set)
        queue = deque((entity, 0, 1.0) for entity in seeds)
        best_entity_score: dict[str, float] = {}
        max_hops = max(0, min(int(hops), 2))

        while queue:
            entity, depth, weight = queue.popleft()
            if weight <= best_entity_score.get(entity, -1.0):
                continue
            best_entity_score[entity] = weight
            for chunk_idx in entity_to_chunks.get(entity, []):
                chunk_scores[int(chunk_idx)] += weight
                matched[int(chunk_idx)].add(entity)
            if depth >= max_hops:
                continue
            for neighbor, edge_weight in neighbors.get(entity, [])[:40]:
                edge_factor = min(1.0, 0.35 + np.log1p(float(edge_weight)) / 5.0)
                queue.append((neighbor, depth + 1, weight * 0.55 * edge_factor))
        return chunk_scores, matched, set(query_entities)

    def retrieve(self, query: str, top_k: int = 5, graph_hops: int = 1):
        query = str(query or "").strip()
        if not query:
            return []
        top_k = max(1, int(top_k))
        q = self.encoder.encode([query], is_query=True, batch_size=1)[0]
        vector_scores = np.asarray(self.embeddings @ q, dtype=np.float32)
        pool = min(len(self.chunks), max(top_k * 8, 32))
        if pool <= 0:
            return []
        order = np.argpartition(-vector_scores, pool - 1)[:pool]
        order = order[np.argsort(-vector_scores[order])]
        vector_order = [int(i) for i in order]

        graph_scores, matched_entities, query_entities = self._graph_scores(query, vector_order, graph_hops)
        max_graph = max(graph_scores.values(), default=1.0) or 1.0

        candidates = set(vector_order)
        # Include graph-only candidates, but cap to avoid huge expansions.
        for idx, _ in sorted(graph_scores.items(), key=lambda item: -item[1])[: pool * 2]:
            candidates.add(int(idx))

        scored = []
        for idx in candidates:
            v = float(vector_scores[idx])
            vector_norm = max(0.0, min(1.0, (v + 1.0) / 2.0))
            graph_norm = float(graph_scores.get(idx, 0.0)) / float(max_graph)
            entities = set(self.chunks[idx].get("entities", []))
            lexical = len(entities.intersection(query_entities)) / max(1, len(query_entities)) if query_entities else 0.0
            final = 0.72 * vector_norm + 0.20 * graph_norm + 0.08 * lexical
            scored.append((final, idx, graph_norm))

        scored.sort(key=lambda item: (-item[0], item[1]))
        hits: list[RetrievalHit] = []
        for final, idx, graph_norm in scored[:top_k]:
            item = self.chunks[idx]
            chunk = DocumentChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                source=item["source"],
                metadata=item.get("metadata") or {},
            )
            matched = tuple(sorted(matched_entities.get(idx, set()))[:12])
            hits.append(RetrievalHit(
                chunk=chunk,
                score=float(final),
                retrieval_path="hybrid" if graph_norm > 0 else "vector",
                graph_entities=matched,
            ))
        return hits


@lru_cache(maxsize=6)
def load_cached_knowledge_base(index_dir: str, fingerprint: str):
    # fingerprint is deliberately part of the cache key so rebuilding an index
    # invalidates the old in-memory graph/embeddings without requiring restart.
    return LocalHybridMedicalKnowledgeBase(index_dir)
