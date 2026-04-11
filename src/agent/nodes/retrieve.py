"""
Retrieve node: top-k similar chunks from the local FAISS RAG index.

Does not require an LLM. If ``data/rag`` is missing, returns empty chunks and
sets ``rag_error`` so downstream nodes can still run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.agent.state import AgentState


def _default_rag_store_dir() -> Path:
    # src/agent/nodes/retrieve.py → four parents up to repo root
    return Path(__file__).resolve().parent.parent.parent.parent / "data" / "rag"


def run_retrieve_node(
    state: AgentState,
    *,
    store_dir: Optional[Path] = None,
    top_k: int = 5,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """
    Embed ``cleaned_text`` (fallback ``raw_text``) and run similarity search.

    Args:
        state: Graph state after ML; uses ``cleaned_text`` for the query string.
        store_dir: Directory with ``faiss.index`` and ``chunks.json``.
        top_k: Number of hits (default 5).
        **_kwargs: Reserved.

    Returns:
        Partial update with ``retrieved_chunks`` (possibly empty) and optional
        ``rag_error`` when the index is absent or retrieval fails.
    """
    base = Path(store_dir) if store_dir is not None else _default_rag_store_dir()
    query = (state.get("cleaned_text") or state.get("raw_text") or "").strip()
    if not query:
        return {
            "retrieved_chunks": [],
            "rag_error": "retrieve: no cleaned_text or raw_text to query.",
        }

    index_path = base / "faiss.index"
    chunks_path = base / "chunks.json"
    if not index_path.is_file() or not chunks_path.is_file():
        return {
            "retrieved_chunks": [],
            "rag_error": (
                f"RAG index not found under {base}. Run: python scripts/build_rag_index.py"
            ),
        }

    try:
        from src.rag.retrieve import retrieve

        hits = retrieve(query, store_dir=base, top_k=top_k)
    except Exception as exc:  # pragma: no cover - defensive
        return {"retrieved_chunks": [], "rag_error": str(exc)}

    return {"retrieved_chunks": hits}
