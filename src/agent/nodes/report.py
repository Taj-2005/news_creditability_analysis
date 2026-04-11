"""
Report node: merge ML outputs (and future RAG / verification) into a final report.

Will produce structured data for Streamlit or export (Markdown / JSON).
No template or LLM logic in this scaffolding phase.
"""

from typing import Any, Dict

from src.agent.state import AgentState


def run_report_node(state: AgentState, **_kwargs: Any) -> Dict[str, Any]:
    """
    Produce a partial state update with ``final_report`` populated.

    Args:
        state: Current graph state; will read ``ml_*`` and optional retrieval
            fields when implemented.
        **_kwargs: Reserved for report format or locale options.

    Returns:
        A mapping containing at least ``final_report`` when implemented.
        Currently returns an empty dict.
    """
    return {}


def describe_report_step() -> str:
    """
    Human-readable description of this node for docs or UI tooltips.

    Returns:
        Short summary string.
    """
    return "Report: synthesize ML (and future evidence) into user-facing output."
