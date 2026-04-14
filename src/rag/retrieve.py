"""
Query the local RAG store: embed query → retrieve relevant chunks.

Supports:
- FAISS store (default): cosine similarity via L2-normalized inner product
- Optional MMR diversification (fetch_k → MMR → top_k)
- Optional Chroma store (if built + installed) for parity with other projects
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from src.rag.embeddings import DEFAULT_MODEL_NAME, EmbeddingModel
from src.rag.store import RAGStore

PathLike = Union[str, Path]
SearchType = Literal["similarity", "mmr"]
BackendType = Literal["faiss", "chroma"]


def _as_float_list(x: Any) -> List[float]:
    return [float(v) for v in (x or [])]


def _mmr_select(
    *,
    query_vec: List[float],
    doc_vecs: List[List[float]],
    top_k: int,
    lambda_mult: float,
) -> List[int]:
    """
    Maximal Marginal Relevance over cosine (vectors are assumed L2-normalized).

    Returns indices into ``doc_vecs``.
    """
    if top_k <= 0 or not doc_vecs:
        return []
    lam = float(lambda_mult)
    lam = 0.6 if not (0.0 <= lam <= 1.0) else lam

    def dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    selected: List[int] = []
    candidates = list(range(len(doc_vecs)))
    while candidates and len(selected) < top_k:
        best_i = candidates[0]
        best_score = -1e9
        for i in candidates:
            sim_to_query = dot(doc_vecs[i], query_vec)
            if not selected:
                score = sim_to_query
            else:
                max_sim_to_selected = max(dot(doc_vecs[i], doc_vecs[j]) for j in selected)
                score = lam * sim_to_query - (1.0 - lam) * max_sim_to_selected
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        candidates.remove(best_i)
    return selected


def retrieve(
    query: str,
    *,
    store_dir: PathLike,
    top_k: int = 5,
    search_type: SearchType = "similarity",
    fetch_k: int = 20,
    lambda_mult: float = 0.6,
    backend: BackendType = "faiss",
    model_name: str = DEFAULT_MODEL_NAME,
    embedding_model: Optional[EmbeddingModel] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve chunks relevant to ``query``.

    Args:
        query: User question or search string (non-empty after strip).
        store_dir: Directory containing ``faiss.index`` and ``chunks.json``.
        top_k: Number of hits (default 5).
        search_type: ``similarity`` (top-k) or ``mmr`` (diversified).
        fetch_k: For ``mmr``, initial candidate pool size before diversification.
        lambda_mult: MMR relevance/diversity tradeoff (0..1).
        backend: ``faiss`` (default) or ``chroma`` (requires optional deps + built store).
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

    model = embedding_model or EmbeddingModel(model_name)
    q_emb = model.encode_query(q)

    if backend == "chroma":
        try:
            from src.rag.chroma_store import chroma_retrieve
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Chroma backend unavailable: {exc}") from exc
        return chroma_retrieve(
            q,
            store_dir=store_dir,
            top_k=top_k,
            search_type=search_type,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            embedding_model=model,
        )

    store = RAGStore.load(Path(store_dir))
    if search_type == "mmr":
        k0 = max(int(fetch_k), int(top_k))
        scores, indices = store.search(q_emb, top_k=k0)
    else:
        scores, indices = store.search(q_emb, top_k=top_k)

    results: List[Dict[str, Any]] = []
    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        if idx < 0:  # FAISS padding when fewer than k results
            continue
        chunk = store.get_chunk(int(idx))
        scored.append(
            (
                float(score),
                int(idx),
                {
                    "text": chunk["text"],
                    "score": float(score),
                    "id": chunk["id"],
                    "metadata": chunk.get("metadata") or {},
                },
            )
        )

    if search_type != "mmr":
        return [x[2] for x in scored]

    # MMR: re-embed the candidate texts (small pool) and diversify.
    cand_texts = [x[2]["text"] for x in scored]
    doc_emb = model.encode(cand_texts, batch_size=16, show_progress_bar=False)
    qv = _as_float_list(q_emb.reshape(-1).tolist())
    dv = [list(map(float, row.tolist())) for row in doc_emb]
    sel = _mmr_select(query_vec=qv, doc_vecs=dv, top_k=top_k, lambda_mult=lambda_mult)
    out = [scored[i][2] for i in sel if 0 <= i < len(scored)]
    # Keep a stable fallback ordering if selection is short.
    if len(out) < top_k:
        used = {id(x) for x in out}
        for _, _, item in scored:
            if id(item) in used:
                continue
            out.append(item)
            if len(out) >= top_k:
                break
    return out
