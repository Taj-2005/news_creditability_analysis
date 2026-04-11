"""
Normalize node: raw user text → cleaned text for downstream ML and RAG.

Delegates to ``clean_text`` in ``src.features.preprocessing`` (same as training
and Streamlit inference).
"""

from typing import Any, Dict

from src.agent.state import AgentState
from src.features.preprocessing import clean_text


def run_normalize_node(state: AgentState, **_kwargs: Any) -> Dict[str, Any]:
    """
    Produce ``cleaned_text`` from ``raw_text`` using shared preprocessing.

    Args:
        state: Must include non-empty ``raw_text`` after strip.
        **_kwargs: Reserved.

    Returns:
        Partial state with ``cleaned_text``, or ``error`` if input is missing.
    """
    raw = (state.get("raw_text") or "").strip()
    if not raw:
        return {
            "error": "normalize: raw_text is required and cannot be empty.",
            "cleaned_text": "",
        }
    return {"cleaned_text": clean_text(raw)}


def describe_normalize_step() -> str:
    """
    Human-readable description of this node for docs or UI tooltips.

    Returns:
        Short summary string.
    """
    return "Normalize: raw input to cleaned text for ML and retrieval."
