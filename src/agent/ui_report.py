"""
Build UI-facing ``final_report`` objects from ML + verification state.

Output shape (exact keys for dashboards):

    summary, risk_factors, fact_checks, verdict, confidence
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# Confidence tiers for display strings
_HIGH = 0.80
_MED = 0.65


def _ml_verdict_label(ml_label: Optional[int]) -> str:
    if ml_label is None:
        return "Unknown"
    return "Fake" if int(ml_label) == 0 else "Real"


def _confidence_band(conf: Optional[float]) -> str:
    if conf is None:
        return "Unavailable"
    c = float(conf)
    if c >= _HIGH:
        return "High"
    if c >= _MED:
        return "Medium"
    return "Low"


def _confidence_display(
    ml_label: Optional[int],
    conf: Optional[float],
    p_fake: Optional[float],
    p_real: Optional[float],
) -> str:
    """Single human-readable confidence line for UI."""
    if conf is None:
        return "Model confidence unavailable."
    band = _confidence_band(conf)
    vlab = _ml_verdict_label(ml_label)
    pct = f"{float(conf):.0%}"
    tail = ""
    if p_fake is not None and p_real is not None:
        tail = f" (P(Fake) {float(p_fake):.0%}, P(Real) {float(p_real):.0%})"
    return f"{band} — {pct} predicted-class probability for “{vlab}”{tail}"


def _fact_checks(verification: Dict[str, Any]) -> List[Dict[str, str]]:
    """Structured rows for list / card UIs."""
    rows: List[Dict[str, str]] = []
    for key, status in (
        ("supported", "supported"),
        ("contradicted", "contradicted"),
        ("unknown", "unknown"),
    ):
        for item in verification.get(key) or []:
            t = str(item).strip()
            if not t:
                continue
            rows.append({"status": status, "finding": t})
    return rows


def _risk_factors(state: Dict[str, Any], verification: Dict[str, Any]) -> List[str]:
    """Ordered bullet strings for risk / caution panels."""
    out: List[str] = []

    if state.get("error"):
        out.append(f"Input or model error: {str(state['error'])[:240]}")

    conf = state.get("ml_confidence")
    if conf is not None and float(conf) < _MED:
        out.append(
            "The credibility model scored this article below the usual confidence bar; "
            "RAG and deeper checks were applied when configured."
        )

    for c in verification.get("contradicted") or []:
        s = str(c).strip()
        if s:
            out.append(f"Evidence tension: {s}")

    n_unk = len(verification.get("unknown") or [])
    if n_unk >= 3:
        out.append("Several aspects could not be matched clearly to retrieved evidence.")

    if state.get("rag_error"):
        out.append(f"Retrieval limitation: {str(state['rag_error'])[:240]}")

    if state.get("llm_query_error"):
        out.append(f"Query planning note: {str(state['llm_query_error'])[:200]}")

    if verification.get("mode") == "fallback" and verification.get("llm_error"):
        out.append(f"LLM verification fallback: {str(verification['llm_error'])[:200]}")

    # Dedupe while preserving order
    seen = set()
    deduped: List[str] = []
    for r in out:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped[:15]


def _deterministic_summary(
    verdict: str,
    confidence_line: str,
    verification: Dict[str, Any],
    rag_used: bool,
    preview: str,
) -> str:
    """Fallback when Groq is unavailable or fails."""
    parts = [
        f"Verdict (ML): {verdict}.",
        f"Confidence: {confidence_line}",
    ]
    if rag_used:
        parts.append(
            "Evidence: Retrieved passages were compared to the article (see fact checks)."
        )
    else:
        parts.append(
            "Evidence: Full RAG path was not run (high model confidence or upstream skip)."
        )
    sup = len(verification.get("supported") or [])
    con = len(verification.get("contradicted") or [])
    unk = len(verification.get("unknown") or [])
    parts.append(
        f"Verification buckets: {sup} supported, {con} tension items, {unk} unclear."
    )
    if preview.strip():
        pv = preview.strip()
        parts.append(
            f"Article preview: {pv[:400]}{'…' if len(pv) > 400 else ''}"
        )
    return "\n\n".join(parts)


def build_ui_final_report(
    state: Dict[str, Any],
    *,
    use_llm_summary: bool = True,
) -> Dict[str, Any]:
    """
    Assemble the canonical UI ``final_report`` dict.

    Args:
        state: Merged LangGraph state (or any dict with the expected keys).
        use_llm_summary: If True, try Groq for ``summary``; else deterministic only.

    Returns:
        Dict with keys ``summary``, ``risk_factors``, ``fact_checks``, ``verdict``,
        ``confidence``, and ``sources`` (RAG snippets for UI citations).
    """
    ml_label = state.get("ml_label")
    verdict = _ml_verdict_label(ml_label)
    conf = state.get("ml_confidence")
    if isinstance(conf, (int, float)):
        conf_f: Optional[float] = float(conf)
    else:
        conf_f = None

    p_fake = state.get("ml_p_fake")
    p_real = state.get("ml_p_real")
    if p_fake is not None:
        p_fake = float(p_fake)
    if p_real is not None:
        p_real = float(p_real)

    confidence = _confidence_display(ml_label, conf_f, p_fake, p_real)
    verification = state.get("verification") or {}
    fact_checks = _fact_checks(verification)
    risk_factors = _risk_factors(state, verification)
    rag_used = "retrieved_chunks" in state

    preview = (state.get("cleaned_text") or state.get("raw_text") or "")[:1500]

    summary = _deterministic_summary(
        verdict, confidence, verification, rag_used, preview
    )

    if use_llm_summary:
        payload = {
            "verdict": verdict,
            "confidence": confidence,
            "risk_factors": risk_factors[:8],
            "fact_checks_preview": fact_checks[:8],
            "rag_path_used": rag_used,
            "verification_mode": verification.get("mode"),
        }
        prompt = (
            "Write a short Summary for a news credibility dashboard (plain text only, "
            "no markdown headings, 2–4 short paragraphs). Base it strictly on the JSON facts; "
            "do not invent URLs or sources. If evidence was weak, say so clearly.\n\n"
            f"FACTS_JSON:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
            f"ARTICLE_SNIPPET:\n{preview[:1200]}\n"
        )
        try:
            from src.agent.llm_service import generate

            llm_text = generate(prompt, temperature=0.3, max_tokens=512).strip()
            if llm_text:
                summary = llm_text
        except Exception:
            pass

    sources: List[Dict[str, Any]] = []
    for i, c in enumerate(state.get("retrieved_chunks") or []):
        if i >= 8:
            break
        txt = str(c.get("text") or "").strip()
        if not txt:
            continue
        sources.append(
            {
                "excerpt": txt[:500] + ("…" if len(txt) > 500 else ""),
                "score": float(c.get("score") or 0.0),
                "metadata": c.get("metadata") or {},
            }
        )

    return {
        "summary": summary,
        "risk_factors": risk_factors,
        "fact_checks": fact_checks,
        "verdict": verdict,
        "confidence": confidence,
        "sources": sources,
    }
