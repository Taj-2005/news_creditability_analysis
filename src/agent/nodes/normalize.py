"""
Normalize node: raw user text → cleaned text for downstream ML and RAG.

Will delegate to shared preprocessing (e.g. ``clean_text``) without
duplicating logic. No implementation in this scaffolding phase.
"""

from typing import Any, Dict

from src.agent.state import AgentState


def run_normalize_node(state: AgentState, **_kwargs: Any) -> Dict[str, Any]:
    """
    Produce a partial state update with ``cleaned_text`` derived from ``raw_text``.

    Args:
        state: Current graph state; must include ``raw_text`` when implemented.
        **_kwargs: Reserved for future dependencies (e.g. custom stopword sets).

    Returns:
        A mapping of state keys to merge into the graph state (e.g.
        ``{"cleaned_text": "..."}``). Currently returns an empty dict.
    """
    return {}


def describe_normalize_step() -> str:
    """
    Human-readable description of this node for docs or UI tooltips.

    Returns:
        Short summary string.
    """
    return "Normalize: raw input to cleaned text for ML and retrieval."
