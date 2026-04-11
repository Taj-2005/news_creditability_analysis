"""
Report node: merge ML, optional RAG hits, and verification into ``final_report``.
"""

from typing import Any, Dict

from src.agent.state import AgentState


def run_report_node(state: AgentState, **_kwargs: Any) -> Dict[str, Any]:
    """
    Build a structured ``final_report`` for UI or export.

    Args:
        state: Full accumulated state from prior nodes.
        **_kwargs: Reserved (e.g. template id).

    Returns:
        Partial update ``{"final_report": {...}}``.
    """
    ml_label = state.get("ml_label")
    if ml_label is None:
        verdict = "Unknown"
    else:
        verdict = "Fake" if int(ml_label) == 0 else "Real"

    conf = state.get("ml_confidence")
    # Retrieve node always emits ``retrieved_chunks`` when it runs (low-confidence path).
    rag_path_used = "retrieved_chunks" in state

    final_report: Dict[str, Any] = {
        "verdict": verdict,
        "ml_label": ml_label,
        "ml_confidence": conf,
        "ml_p_fake": state.get("ml_p_fake"),
        "ml_p_real": state.get("ml_p_real"),
        "cleaned_text_preview": (state.get("cleaned_text") or "")[:500],
        "retrieved_chunks": state.get("retrieved_chunks") or [],
        "verification": state.get("verification") or {},
        "error": state.get("error"),
        "rag_error": state.get("rag_error"),
        "rag_path_used": rag_path_used,
    }
    return {"final_report": final_report}


def describe_report_step() -> str:
    """
    Human-readable description of this node for docs or UI tooltips.

    Returns:
        Short summary string.
    """
    return "Report: synthesize ML (and future evidence) into user-facing output."
