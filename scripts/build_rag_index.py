#!/usr/bin/env python3
"""
Build a small local FAISS RAG index (MiniLM embeddings).

Usage (from repo root):
  python scripts/build_rag_index.py

Outputs under data/rag/:
  faiss.index, chunks.json

Uses a tiny in-repo sample corpus (mock-style news snippets); safe to run offline
after ``pip install -r requirements.txt``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.embeddings import EmbeddingModel
from src.rag.store import RAGStore

# --- Sample corpus (short, self-contained; not from Kaggle to keep repo tiny) ---
SAMPLE_DOCUMENTS: List[str] = [
    (
        "Federal Reserve officials held rates steady citing inflation trends and labor market "
        "balance. Analysts expect gradual adjustments over the next quarters."
    ),
    (
        "Clinical trial results published in a peer-reviewed journal showed improved outcomes "
        "for patients receiving the standard-of-care protocol versus placebo."
    ),
    (
        "City council approved a budget increase for public transit maintenance and safety "
        "upgrades scheduled to begin next fiscal year."
    ),
    (
        "BREAKING YOU WONT BELIEVE miracle cure doctors hate this one trick share before deleted "
        "click now for secret government cover up exposed."
    ),
    (
        "Scientists warn that extraordinary claims require rigorous replication before policy "
        "recommendations can be considered credible."
    ),
    (
        "Weather service issued a routine advisory for coastal areas with minor flooding possible "
        "during high tide; residents should follow local guidance."
    ),
    (
        "Celebrity death hoax spreads on social media using sensational headlines and urgent calls "
        "to share; fact-checkers recommend verifying primary sources."
    ),
    (
        "Trade negotiators concluded a technical round on tariff schedules with further meetings "
        "planned; no final agreement was announced."
    ),
]


def chunk_text(
    text: str,
    *,
    max_chars: int = 400,
    overlap: int = 60,
) -> List[str]:
    """
    Simple character-window chunking with overlap.

    Short texts yield a single chunk.
    """
    t = " ".join((text or "").split())
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    chunks: List[str] = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(t):
        piece = t[start : start + max_chars].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def build_chunks(
    documents: List[str],
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Flatten documents into chunk strings with doc_index metadata."""
    texts: List[str] = []
    metas: List[Dict[str, object]] = []
    for doc_i, doc in enumerate(documents):
        for part_i, chunk in enumerate(chunk_text(doc)):
            texts.append(chunk)
            metas.append({"doc_index": doc_i, "chunk_index": part_i})
    return texts, metas


def main() -> int:
    out_dir = REPO_ROOT / "data" / "rag"
    print(f"Building RAG index → {out_dir}")

    chunk_texts, metadatas = build_chunks(SAMPLE_DOCUMENTS)
    if not chunk_texts:
        print("No chunks produced.", file=sys.stderr)
        return 1

    print(f"Chunks: {len(chunk_texts)} (from {len(SAMPLE_DOCUMENTS)} sample docs)")

    model = EmbeddingModel()
    embeddings = model.encode(chunk_texts, batch_size=16, show_progress_bar=False)

    store = RAGStore(model.dimension)
    store.add(embeddings, chunk_texts, metadatas=metadatas)
    store.save(out_dir)

    print(f"Saved FAISS index ({store.n_chunks} vectors, dim={store.dimension})")

    # Smoke retrieval
    from src.rag.retrieve import retrieve

    demo_query = "interest rates inflation federal reserve"
    hits = retrieve(demo_query, store_dir=out_dir, top_k=5, embedding_model=model)
    print("\nSmoke query:", demo_query)
    print(json.dumps(hits[:3], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
