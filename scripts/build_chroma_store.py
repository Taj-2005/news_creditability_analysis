#!/usr/bin/env python3
"""
Build an optional Chroma store for the news RAG knowledge base.

This is not required for the default FAISS pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.chroma_store import build_chroma_store
from src.rag.embeddings import EmbeddingModel


def main() -> int:
    kb_dir = REPO_ROOT / "data" / "rag" / "knowledge_base"
    out_base = REPO_ROOT / "data" / "rag"
    print("Building Chroma store from:", kb_dir)
    out = build_chroma_store(kb_dir=kb_dir, out_base_dir=out_base, embedding_model=EmbeddingModel())
    print("Saved Chroma store to:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

