"""
Verify node: align claims with retrieved evidence using Groq (with graceful fallback).
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agent.state import AgentState


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _format_chunks(chunks: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    parts: List[str] = []
    n = 0
    for i, c in enumerate(chunks):
        body = (c.get("text") or "").strip()
        if not body:
            continue
        meta = c.get("metadata") or {}
        sc = c.get("score", 0)
        try:
            scf = float(sc)
        except (TypeError, ValueError):
            scf = 0.0
        header = f"[{i + 1}] score={scf:.3f} meta={meta}\n"
        block = header + body
        if n + len(block) > max_chars:
            break
        parts.append(block)
        n += len(block)
    return "\n\n".join(parts) if parts else "(no chunk text)"


def run_verify_node(state: AgentState, **_kwargs: Any) -> Dict[str, Any]:
    """
    Produce a ``verification`` dict; uses LLM when ``GROQ_API_KEY`` is available.

    On LLM failure, returns structured placeholder data plus ``llm_error``.
    """
    chunks = state.get("retrieved_chunks") or []
    rag_err = state.get("rag_error")
    excerpt = (state.get("cleaned_text") or state.get("raw_text") or "").strip()[:4000]

    base: Dict[str, Any] = {
        "chunks_reviewed": len(chunks),
        "top_scores": [_safe_float(c.get("score")) for c in chunks[:3]],
    }
    if rag_err:
        base["rag_error"] = rag_err

    if not chunks:
        base["mode"] = "no_evidence"
        base["llm"] = False
        base["notes"] = "No retrieved passages to verify against."
        return {"verification": base}

    evidence = _format_chunks(chunks)
    ml_line = ""
    if state.get("ml_label") is not None:
        lab = "Fake" if int(state["ml_label"]) == 0 else "Real"
        ml_line = f"ML classifier says: {lab} (confidence {state.get('ml_confidence', 0):.2f}).\n"

    prompt = (
        "You are a careful news credibility assistant. Compare the ARTICLE EXCERPT to the EVIDENCE "
        "snippets (from an external corpus; they may be only loosely related).\n"
        f"{ml_line}\n"
        "Answer in plain text with these sections exactly:\n"
        "SUPPORT: (one short paragraph — what in the evidence supports or aligns with the excerpt, if anything)\n"
        "CONTRADICT: (one short paragraph — what contradicts or undermines it, if anything)\n"
        "VERDICT: (one line: supported | contradicted | unclear | insufficient_evidence)\n"
        "\nARTICLE EXCERPT:\n"
        f"{excerpt}\n\nEVIDENCE:\n{evidence}\n"
    )

    try:
        from src.agent.llm_service import generate

        analysis = generate(prompt)
        base["mode"] = "llm"
        base["llm"] = True
        base["analysis"] = analysis
        return {"verification": base}
    except Exception as exc:
        base["mode"] = "fallback"
        base["llm"] = False
        base["llm_error"] = str(exc)
        base["notes"] = (
            "Groq verification unavailable; see retrieved_chunks in the final report for raw evidence."
        )
        return {"verification": base}
