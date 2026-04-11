"""
Query the local FAISS RAG store: embed question → top-k similar chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.rag.embeddings import DEFAULT_MODEL_NAME, EmbeddingModel
from src.rag.store import RAGStore

PathLike = Union[str, Path]


def retrieve(
    query: str,
    *,
    store_dir: PathLike,
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
    embedding_model: Optional[EmbeddingModel] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k chunks most similar to ``query`` (cosine via normalized IP).

    Args:
        query: User question or search string (non-empty after strip).
        store_dir: Directory containing ``faiss.index`` and ``chunks.json``.
        top_k: Number of hits (default 5); capped by index size.
        model_name: sentence-transformers model id (must match build-time model).
        embedding_model: Optional pre-built ``EmbeddingModel`` to avoid reload.

    Returns:
        List of dicts, each with keys ``text``, ``score``, ``id``, ``metadata``,
        sorted by descending similarity score.

    Raises:
        FileNotFoundError: If the store directory is incomplete.
        ValueError: If ``query`` is empty.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query must be non-empty")

    store = RAGStore.load(Path(store_dir))
    model = embedding_model or EmbeddingModel(model_name)
    q_emb = model.encode_query(q)
    scores, indices = store.search(q_emb, top_k=top_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        if idx < 0:  # FAISS padding when fewer than k results
            continue
        chunk = store.get_chunk(int(idx))
        results.append(
            {
                "text": chunk["text"],
                "score": float(score),
                "id": chunk["id"],
                "metadata": chunk.get("metadata") or {},
            }
        )
    return results
