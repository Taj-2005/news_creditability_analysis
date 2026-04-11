"""
Lightweight local RAG: MiniLM embeddings + FAISS flat index.

Build index with ``python scripts/build_rag_index.py``; retrieve via
``src.rag.retrieve.retrieve``.
"""

from src.rag.embeddings import EmbeddingModel, encode_query, encode_documents
from src.rag.retrieve import retrieve

__all__ = [
    "EmbeddingModel",
    "encode_documents",
    "encode_query",
    "retrieve",
]
