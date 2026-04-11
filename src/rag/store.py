"""
FAISS-backed vector store for chunk embeddings (cosine via normalized inner product).

Persists:
  - ``faiss.index`` — ``IndexFlatIP`` on normalized 384-d vectors
  - ``chunks.json`` — list of ``{"id": int, "text": str, "metadata": dict}``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np


INDEX_FILENAME = "faiss.index"
CHUNKS_FILENAME = "chunks.json"


class RAGStore:
    """
    In-memory FAISS index + chunk payloads with save/load to a directory.
    """

    def __init__(self, dimension: int):
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: List[Dict[str, Any]] = []

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    def add(
        self,
        embeddings: np.ndarray,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """
        Add normalized embedding rows and matching chunk texts.

        Args:
            embeddings: Float32 array ``(n, dimension)``, L2-normalized rows.
            texts: ``n`` chunk strings (same order as rows).
            metadatas: Optional per-chunk metadata dicts.

        Raises:
            ValueError: On shape / length mismatch or wrong dtype/dim.
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        n = embeddings.shape[0]
        if n == 0:
            return
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings dim {embeddings.shape[1]} != store dimension {self.dimension}"
            )
        if len(texts) != n:
            raise ValueError("len(texts) must match number of embedding rows")
        if metadatas is not None and len(metadatas) != n:
            raise ValueError("len(metadatas) must match number of embedding rows")

        base_id = len(self._chunks)
        for i in range(n):
            if metadatas is None:
                meta: Dict[str, Any] = {}
            else:
                meta = dict(metadatas[i] or {})
            self._chunks.append(
                {
                    "id": base_id + i,
                    "text": texts[i],
                    "metadata": meta,
                }
            )
        self._index.add(embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search by a single normalized query vector ``(1, dimension)``.

        Returns:
            Tuple ``(scores, indices)`` — scores are inner product (~cosine), indices into chunks.
        """
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape[1] != self.dimension:
            raise ValueError("query_embedding dimension mismatch")
        k = min(top_k, self.n_chunks)
        if k == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        faiss.normalize_L2(query_embedding)
        scores, indices = self._index.search(query_embedding, k)
        return scores[0], indices[0]

    def get_chunk(self, index: int) -> Dict[str, Any]:
        """Return chunk record by row index (0 .. n_chunks-1)."""
        return dict(self._chunks[int(index)])

    def save(self, directory: Path) -> None:
        """Write index + chunks JSON under ``directory`` (created if missing)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / INDEX_FILENAME))
        with open(directory / CHUNKS_FILENAME, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: Path) -> "RAGStore":
        """Load store from disk."""
        directory = Path(directory)
        index_path = directory / INDEX_FILENAME
        chunks_path = directory / CHUNKS_FILENAME
        if not index_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(
                f"Missing {INDEX_FILENAME} or {CHUNKS_FILENAME} under {directory}"
            )
        with open(chunks_path, encoding="utf-8") as f:
            chunks: List[Dict[str, Any]] = json.load(f)
        index = faiss.read_index(str(index_path))
        dim = index.d
        store = cls(dim)
        store._index = index
        store._chunks = chunks
        return store
