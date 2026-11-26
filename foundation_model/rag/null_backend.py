"""Knowledge-base placeholder used until document/index construction is enabled."""


class KnowledgeBaseNotConfigured(RuntimeError):
    pass


class NullMedicalKnowledgeBase:
    @property
    def ready(self):
        return False

    def retrieve(self, query, top_k=5, graph_hops=1):
        raise KnowledgeBaseNotConfigured(
            "Medical knowledge base has not been built yet. The Hybrid GraphRAG interfaces are ready, "
            "but document ingestion/vector indexing/graph construction are intentionally deferred."
        )
