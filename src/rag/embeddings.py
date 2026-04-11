"""
Sentence-transformers embeddings (MiniLM) for RAG.

Uses ``all-MiniLM-L6-v2`` (384 dimensions) — small, fast, CPU-friendly.
Vectors are L2-normalized for cosine similarity via inner product (FAISS).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

# Short HF id — MiniLM L6, 384-d, CPU-friendly
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _ensure_local_hf_cache() -> None:
    """
    Prefer a repo-local Hugging Face cache so CI/sandboxes without ~/.cache
    write access can still download the model once.
    """
    cache_root = _repo_root() / ".cache" / "huggingface"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    _ensure_local_hf_cache()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingModel:
    """
    Lazy-loaded MiniLM encoder with normalized outputs for FAISS IP search.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._st = None

    @property
    def sentence_transformer(self):
        if self._st is None:
            self._st = _load_sentence_transformer(self.model_name)
        return self._st

    @property
    def dimension(self) -> int:
        st = self.sentence_transformer
        getter = getattr(st, "get_embedding_dimension", None) or getattr(
            st, "get_sentence_embedding_dimension"
        )
        return int(getter())

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode strings to L2-normalized float32 vectors (one row per text).

        Args:
            texts: Non-empty strings to embed.
            batch_size: Encoder batch size.
            show_progress_bar: Passed to sentence-transformers.

        Returns:
            Array of shape ``(len(texts), dimension)``, float32, L2-normalized.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        raw = self.sentence_transformer.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        emb = np.asarray(raw, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        faiss.normalize_L2(emb)
        return emb

    def encode_query(self, text: str) -> np.ndarray:
        """Single query → shape ``(1, dimension)`` normalized."""
        t = (text or "").strip()
        if not t:
            raise ValueError("Query text must be non-empty.")
        return self.encode([t], batch_size=1)


def encode_documents(
    texts: Sequence[str],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """
    Encode a document chunk list (convenience for scripts without a model instance).

    Args:
        texts: Chunk strings.
        model_name: sentence-transformers model id.
        batch_size: Batch size for encoding.
        show_progress_bar: Whether to show tqdm bar.

    Returns:
        Normalized float32 matrix ``(n, dim)``.
    """
    return EmbeddingModel(model_name).encode(
        texts, batch_size=batch_size, show_progress_bar=show_progress_bar
    )


def encode_query(text: str, *, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Encode a single query string; returns shape ``(1, dim)``."""
    return EmbeddingModel(model_name).encode_query(text)
