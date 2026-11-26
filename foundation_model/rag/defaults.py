"""Portable defaults for local medical knowledge-base resources.

No model path is tied to a developer machine. The optional environment variable
can be set by an operator, or a directory can be selected in the Streamlit UI.
"""

import os


def default_local_embedding_model() -> str:
    return os.environ.get("MEDICAL_RAG_EMBEDDING_MODEL", "").strip()
