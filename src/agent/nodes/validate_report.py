"""
Validator node: check ``final_report`` schema and allow a bounded retry loop.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agent.state import AgentState


_REQ_KEYS = (
    "summary",
    "risk_factors",
    "fact_checks",
    "verdict",
    "confidence",
    "sources",
    "credibility_score",
    "pattern_detection_summary",
    "disclaimer",
)


def run_validate_report_node(state: AgentState, **_kwargs: Any) -> Dict[str, Any]:
    fr = state.get("final_report") or {}
    errs: List[str] = []

    for k in _REQ_KEYS:
        if k not in fr:
            errs.append(f"missing:{k}")

    if fr.get("credibility_score") not in ("High", "Low"):
        errs.append("credibility_score_invalid")

    # Basic type/shape checks used by UI.
    if not isinstance(fr.get("risk_factors"), list):
        errs.append("risk_factors_not_list")
    if not isinstance(fr.get("fact_checks"), list):
        errs.append("fact_checks_not_list")
    if not isinstance(fr.get("sources"), list):
        errs.append("sources_not_list")

    summary = str(fr.get("summary") or "").strip()
    if len(summary) < 20:
        errs.append("summary_too_short")

    attempt = int(state.get("report_attempt") or 0)
    passed = len(errs) == 0
    out: Dict[str, Any] = {
        "validation_passed": passed,
        "validation_errors": errs[:10],
        "report_attempt": attempt + (0 if passed else 1),
    }
    return out


def describe_validate_report_step() -> str:
    return "Validate: enforce final_report schema (bounded retry)."

