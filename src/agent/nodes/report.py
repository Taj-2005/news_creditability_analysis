"""
Report node: merge ML, RAG, and verification; optional Groq narrative summary.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from src.agent.state import AgentState


def run_report_node(state: AgentState, **_kwargs: Any) -> Dict[str, Any]:
    """
    Build ``final_report`` and optionally append an LLM-generated narrative summary.
    """
    ml_label = state.get("ml_label")
    if ml_label is None:
        verdict = "Unknown"
    else:
        verdict = "Fake" if int(ml_label) == 0 else "Real"

    conf = state.get("ml_confidence")
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
        "queries": state.get("queries") or [],
        "error": state.get("error"),
        "rag_error": state.get("rag_error"),
        "llm_query_error": state.get("llm_query_error"),
        "rag_path_used": rag_path_used,
    }

    # --- LLM narrative (Groq) for any path that reached report ---
    ver = state.get("verification") or {}
    summary_payload = {
        "verdict": verdict,
        "ml_confidence": conf,
        "rag_path_used": rag_path_used,
        "verification_mode": ver.get("mode"),
        "top_query": (state.get("queries") or [None])[0],
        "verification_claims": {
            "supported": (ver.get("supported") or [])[:5],
            "contradicted": (ver.get("contradicted") or [])[:5],
            "unknown": (ver.get("unknown") or [])[:5],
        },
    }
    prompt = (
        "Write a concise credibility briefing (3–5 short paragraphs) for a technical reader. "
        "Use only the JSON facts below; do not invent sources. Mention uncertainty if RAG was not used.\n\n"
        f"FACTS_JSON:\n{json.dumps(summary_payload, indent=2)}\n\n"
        "ARTICLE_PREVIEW:\n"
        f"{(state.get('cleaned_text') or state.get('raw_text') or '')[:1200]}\n"
    )
    vc = summary_payload["verification_claims"]
    if any(vc.get(k) for k in ("supported", "contradicted", "unknown")):
        prompt += "\nVERIFICATION_CLAIMS_JSON:\n" + json.dumps(vc, ensure_ascii=False) + "\n"

    try:
        from src.agent.llm_service import generate

        final_report["llm_summary"] = generate(prompt)
        final_report["llm_report_error"] = None
    except Exception as exc:
        final_report["llm_summary"] = None
        final_report["llm_report_error"] = str(exc)

    return {"final_report": final_report}


def describe_report_step() -> str:
    """
    Human-readable description of this node for docs or UI tooltips.

    Returns:
        Short summary string.
    """
    return "Report: synthesize ML (and future evidence) into user-facing output."
