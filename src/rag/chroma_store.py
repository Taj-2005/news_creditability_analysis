"""
Optional Chroma-backed vector store for the RAG knowledge base.

This is implemented for parity with other projects. The default runtime continues
to use FAISS (``src.rag.store``) unless the caller requests ``backend="chroma"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from src.rag.embeddings import EmbeddingModel

PathLike = Union[str, Path]
SearchType = Literal["similarity", "mmr"]


CHROMA_DIRNAME = "chroma_store"
COLLECTION_NAME = "news_credibility_kb"


def _store_dir(base: PathLike) -> Path:
    return Path(base) / CHROMA_DIRNAME


def build_chroma_store(
    *,
    kb_dir: PathLike,
    out_base_dir: PathLike,
    embedding_model: Optional[EmbeddingModel] = None,
) -> Path:
    """
    Build a persistent Chroma store from markdown files in ``kb_dir``.

    Stores documents with ids and metadata; embeddings are computed via MiniLM.
    """
    kb_dir = Path(kb_dir)
    out_dir = _store_dir(out_base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted([p for p in kb_dir.rglob("*.md") if p.is_file()])
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    ids: List[str] = []
    for i, p in enumerate(md_files):
        if p.name.lower() == "readme.md":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append(text)
        metas.append({"source_file": p.name, "source_rel": str(p.relative_to(kb_dir.parent.parent.parent))})
        ids.append(f"kb_{i}_{p.stem}")

    model = embedding_model or EmbeddingModel()
    emb = model.encode(docs, batch_size=16, show_progress_bar=False)

    import chromadb  # optional dependency

    client = chromadb.PersistentClient(path=str(out_dir))
    col = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    # Rebuild collection deterministically.
    try:
        col.delete(where={})
    except Exception:
        pass

    col.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=[row.tolist() for row in emb],
    )

    # Write a tiny manifest for debugging.
    (out_dir / "manifest.json").write_text(
        json.dumps({"collection": COLLECTION_NAME, "n_docs": len(docs)}, indent=2),
        encoding="utf-8",
    )
    return out_dir


def chroma_retrieve(
    query: str,
    *,
    store_dir: PathLike,
    top_k: int,
    search_type: SearchType,
    fetch_k: int,
    lambda_mult: float,
    embedding_model: EmbeddingModel,
) -> List[Dict[str, Any]]:
    """
    Retrieve from Chroma. MMR is implemented client-side over a fetched pool.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query must be non-empty")

    import chromadb  # optional dependency

    base = _store_dir(store_dir)
    if not base.exists():
        raise FileNotFoundError(f"Chroma store missing at {base}. Build it first.")

    client = chromadb.PersistentClient(path=str(base))
    col = client.get_collection(name=COLLECTION_NAME)

    q_emb = embedding_model.encode_query(q).reshape(-1).tolist()
    n = max(int(fetch_k), int(top_k)) if search_type == "mmr" else int(top_k)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    scored: List[Dict[str, Any]] = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        # For cosine, chroma distances are typically (1 - cosine_similarity).
        score = 1.0 - float(dist) if dist is not None else 0.0
        scored.append({"text": str(doc), "score": score, "id": i, "metadata": meta or {}})

    if search_type != "mmr":
        return scored[:top_k]

    # MMR over candidate embeddings (re-embed the docs for simplicity).
    doc_emb = embedding_model.encode([s["text"] for s in scored], batch_size=16, show_progress_bar=False)
    qv = list(map(float, q_emb))
    dv = [list(map(float, row.tolist())) for row in doc_emb]

    from src.rag.retrieve import _mmr_select  # reuse implementation

    sel = _mmr_select(query_vec=qv, doc_vecs=dv, top_k=top_k, lambda_mult=lambda_mult)
    return [scored[i] for i in sel if 0 <= i < len(scored)]

