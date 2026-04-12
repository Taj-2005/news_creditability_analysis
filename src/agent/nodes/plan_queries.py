"""
Plan search queries for RAG using the shared LLM service (Groq).

Runs on the low-confidence path before ``retrieve``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agent.state import AgentState


def _fallback_queries(state: AgentState) -> List[str]:
    base = (state.get("cleaned_text") or state.get("raw_text") or "").strip()
    if not base:
        return []
    # Single window as a minimal fallback
    snippet = base[:400]
    return [snippet] if snippet else []


def run_plan_queries_node(state: AgentState, **_kwargs: Any) -> Dict[str, Any]:
    """
    Ask the LLM for short search queries; fall back to a text window on failure.

    Writes ``queries`` (non-empty list) for ``run_retrieve_node`` to consume.
    """
    text = (state.get("cleaned_text") or state.get("raw_text") or "").strip()
    if not text:
        return {"queries": [], "llm_query_error": "plan_queries: no text in state."}

    ml_hint = ""
    if state.get("ml_label") is not None and state.get("ml_confidence") is not None:
        lab = "Fake" if int(state["ml_label"]) == 0 else "Real"
        ml_hint = f"Classifier hint: {lab} (confidence {float(state['ml_confidence']):.2f}).\n"

    prompt = (
        "You help retrieve supporting evidence from a news knowledge base.\n"
        f"{ml_hint}"
        "Given the excerpt below, output exactly 3 short search queries (keywords or short phrases), "
        "one per line, no numbering or bullets. Queries must be in English.\n\n"
        "EXCERPT:\n"
        f"{text[:3000]}\n"
    )

    try:
        from src.agent.llm_service import generate

        raw = generate(prompt)
    except Exception as exc:
        fb = _fallback_queries(state)
        return {
            "queries": fb if fb else [text[:300]],
            "llm_query_error": str(exc),
        }

    lines = [ln.strip().lstrip("-•* ") for ln in raw.splitlines() if ln.strip()]
    queries = [ln for ln in lines if ln][:5]
    if not queries:
        fb = _fallback_queries(state)
        queries = fb if fb else [text[:300]]

    out: Dict[str, Any] = {"queries": queries}
    return out
