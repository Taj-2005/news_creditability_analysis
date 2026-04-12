"""
Report node: UI-facing ``final_report`` from ML + verification (+ optional Groq summary).
"""

from __future__ import annotations

from typing import Any, Dict

from src.agent.state import AgentState
from src.agent.ui_report import build_ui_final_report


def run_report_node(state: AgentState, **_kwargs: Any) -> Dict[str, Any]:
    """
    Build ``final_report`` for dashboards and APIs.

    Output keys (only):

        summary: str — narrative (Groq when available, else deterministic).
        risk_factors: list[str] — cautions for the reader.
        fact_checks: list[dict] — ``{ "status", "finding" }`` rows for list/card UIs.
        verdict: str — ``Fake`` | ``Real`` | ``Unknown`` (from ML label).
        confidence: str — human-readable confidence line.

    Combines ``ml_*`` fields and ``verification`` (``supported`` / ``contradicted`` / ``unknown``).
    """
    # TypedDict is dict-like; pass through for the builder
    final_report = build_ui_final_report(dict(state), use_llm_summary=True)
    return {"final_report": final_report}


def describe_report_step() -> str:
    """
    Human-readable description of this node for docs or UI tooltips.

    Returns:
        Short summary string.
    """
    return "Report: synthesize ML (and future evidence) into user-facing output."
