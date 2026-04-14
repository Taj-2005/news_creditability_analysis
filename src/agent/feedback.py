"""
Local feedback loop for the credibility agent.

Stores user ratings as JSONL under ``data/feedback/feedback.jsonl``.
This is optional and does not affect core inference unless integrated by a caller.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _feedback_path() -> Path:
    base = _repo_root() / "data" / "feedback"
    base.mkdir(parents=True, exist_ok=True)
    return base / "feedback.jsonl"


def record_feedback(
    *,
    raw_text: str,
    verdict: str,
    credibility_score: str,
    rating: int,
    notes: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Append a feedback row.

    Args:
        raw_text: User-provided article text (stored trimmed).
        verdict: Fake/Real/Unknown.
        credibility_score: High/Low.
        rating: 1..5 (UI scale).
        notes: Optional user notes.
        metadata: Optional structured context (page, model, etc.).

    Returns:
        Path to the JSONL file.
    """
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "verdict": str(verdict),
        "credibility_score": str(credibility_score),
        "rating": int(rating),
        "notes": (notes or "").strip()[:2000],
        "raw_text_preview": (raw_text or "").strip()[:800],
        "metadata": dict(metadata or {}),
    }
    p = _feedback_path()
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p

