"""
Retrieve node: top-k similar chunks from the local RAG store.

Default backend is **FAISS** (``faiss.index`` + ``chunks.json``). Optional backend is **Chroma**
(``data/rag/chroma_store/``) when built via ``scripts/build_chroma_store.py``.

Does not require an LLM. If the selected store is missing, returns empty chunks and sets
``rag_error`` so downstream nodes can still run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.agent.state import AgentState


def _default_rag_store_dir() -> Path:
    # src/agent/nodes/retrieve.py → four parents up to repo root
    return Path(__file__).resolve().parent.parent.parent.parent / "data" / "rag"


def build_retrieval_query(state: AgentState) -> str:
    """
    Prefer LLM-planned ``queries`` (joined); else fall back to cleaned/raw text.
    """
    queries = state.get("queries")
    if isinstance(queries, list) and queries:
        parts = [str(q).strip() for q in queries[:4] if str(q).strip()]
        if parts:
            return " ".join(parts)
    return (state.get("cleaned_text") or state.get("raw_text") or "").strip()


def run_retrieve_node(
    state: AgentState,
    *,
    store_dir: Optional[Path] = None,
    top_k: int = 5,
    backend: str = "faiss",
    search_type: str = "similarity",
    fetch_k: int = 20,
    lambda_mult: float = 0.6,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """
    Embed the retrieval query (planned queries or cleaned text) and search the RAG backend.

    Args:
        state: Graph state; uses ``queries`` from ``plan_queries`` when present.
        store_dir: RAG directory (FAISS files live here; Chroma uses ``<store_dir>/chroma_store``).
        top_k: Number of hits (default 5).
        backend: ``faiss`` or ``chroma``.
        search_type: ``similarity`` or ``mmr``.
        fetch_k: Candidate pool size for MMR.
        lambda_mult: MMR tradeoff (0..1).
        **_kwargs: Reserved.

    Returns:
        Partial update with ``retrieved_chunks`` (possibly empty) and optional
        ``rag_error`` when the index is absent or retrieval fails.
    """
    base = Path(store_dir) if store_dir is not None else _default_rag_store_dir()
    query = build_retrieval_query(state)
    if not query:
        return {
            "retrieved_chunks": [],
            "rag_error": "retrieve: no cleaned_text or raw_text to query.",
        }

    if backend == "chroma":
        chroma_dir = base / "chroma_store"
        if not chroma_dir.is_dir():
            return {
                "retrieved_chunks": [],
                "rag_error": (
                    "Chroma store missing (expected data/rag/chroma_store/). "
                    "From the repo root run: python scripts/build_chroma_store.py — then commit "
                    "data/rag/chroma_store/ for Streamlit Cloud."
                ),
            }
    else:
        index_path = base / "faiss.index"
        chunks_path = base / "chunks.json"
        if not index_path.is_file() or not chunks_path.is_file():
            return {
                "retrieved_chunks": [],
                "rag_error": (
                    "RAG index missing (expected data/rag/faiss.index and data/rag/chunks.json). "
                    "From the repo root run: python scripts/build_rag_index.py — then commit "
                    "data/rag/ for Streamlit Cloud, which does not run that build automatically."
                ),
            }

    try:
        from src.rag.retrieve import retrieve

        hits = retrieve(
            query,
            store_dir=base,
            top_k=top_k,
            backend="chroma" if backend == "chroma" else "faiss",
            search_type="mmr" if search_type == "mmr" else "similarity",
            fetch_k=int(fetch_k),
            lambda_mult=float(lambda_mult),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"retrieved_chunks": [], "rag_error": str(exc)}

    return {
        "retrieved_chunks": hits,
        "rag_backend": "chroma" if backend == "chroma" else "faiss",
        "rag_search_type": "mmr" if search_type == "mmr" else "similarity",
        "rag_fetch_k": int(fetch_k),
        "rag_lambda_mult": float(lambda_mult),
    }
