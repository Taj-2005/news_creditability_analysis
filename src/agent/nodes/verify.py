"""
Verification node: compare article text to retrieved chunks via LLM → structured JSON.

Output shape (always present after this node):

- ``supported``: list of short strings — claims aligned with evidence
- ``contradicted``: list of short strings — claims in tension with evidence
- ``unknown``: list of short strings — unclear or insufficient to judge

Parsing is strict: invalid model output is replaced with a deterministic fallback
so downstream nodes always receive clean lists.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agent.state import AgentState

_JSON_KEYS: Tuple[str, str, str] = ("supported", "contradicted", "unknown")
_MAX_ITEMS_PER_BUCKET = 8
_MAX_STRING_LEN = 120


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
        scf = _safe_float(c.get("score"))
        header = f"[{i + 1}] score={scf:.3f} meta={meta}\n"
        block = header + body
        if n + len(block) > max_chars:
            break
        parts.append(block)
        n += len(block)
    return "\n\n".join(parts) if parts else "(no chunk text)"


def _strip_json_fences(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _clean_string_item(s: Any) -> str:
    t = str(s).strip().replace("\n", " ")
    if len(t) > _MAX_STRING_LEN:
        t = t[: _MAX_STRING_LEN - 1] + "…"
    return t


def _normalize_bucket_lists(obj: Any) -> Dict[str, List[str]]:
    """Return only ``supported`` / ``contradicted`` / ``unknown`` as deduped string lists."""
    empty = {k: [] for k in _JSON_KEYS}
    if not isinstance(obj, dict):
        return empty
    out: Dict[str, List[str]] = {k: [] for k in _JSON_KEYS}
    for key in _JSON_KEYS:
        raw = obj.get(key)
        items: List[str] = []
        if raw is None:
            pass
        elif isinstance(raw, str) and raw.strip():
            items = [_clean_string_item(raw)]
        elif isinstance(raw, list):
            for x in raw:
                c = _clean_string_item(x)
                if c:
                    items.append(c)
        else:
            c = _clean_string_item(raw)
            if c:
                items.append(c)
        seen: set[str] = set()
        for it in items[:_MAX_ITEMS_PER_BUCKET]:
            if it not in seen:
                seen.add(it)
                out[key].append(it)
    return out


def _parse_llm_verification_json(raw: str) -> Dict[str, List[str]]:
    """Parse model output into three buckets; raises on unusable input."""
    cleaned = _strip_json_fences(raw)
    data = json.loads(cleaned)
    return _normalize_bucket_lists(data)


def _empty_structured(
    *,
    mode: str,
    llm: bool,
    unknown_note: str,
    chunks_reviewed: int = 0,
    top_scores: Optional[List[float]] = None,
    rag_error: Optional[str] = None,
    llm_error: Optional[str] = None,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "supported": [],
        "contradicted": [],
        "unknown": ([unknown_note] if unknown_note else []),
        "mode": mode,
        "llm": llm,
        "chunks_reviewed": chunks_reviewed,
        "top_scores": top_scores or [],
    }
    if rag_error:
        base["rag_error"] = rag_error
    if llm_error:
        base["llm_error"] = llm_error
    return base


def run_verify_node(state: AgentState, **_kwargs: Any) -> Dict[str, Any]:
    """
    Compare article (cleaned/raw) to retrieved chunks using Groq; emit structured JSON lists.

    Returns:
        Partial state update ``{"verification": {...}}`` with keys ``supported``,
        ``contradicted``, ``unknown`` (lists of strings) plus metadata.
    """
    chunks = state.get("retrieved_chunks") or []
    rag_err = state.get("rag_error")
    article = (state.get("cleaned_text") or state.get("raw_text") or "").strip()
    excerpt = article[:4000]
    top_scores = [_safe_float(c.get("score")) for c in chunks[:3]]

    if not chunks:
        return {
            "verification": _empty_structured(
                mode="no_evidence",
                llm=False,
                unknown_note="No retrieved passages to verify against.",
                chunks_reviewed=0,
                top_scores=top_scores,
                rag_error=rag_err,
            )
        }

    evidence = _format_chunks(chunks)
    ml_hint = ""
    if state.get("ml_label") is not None:
        lab = "Fake" if int(state["ml_label"]) == 0 else "Real"
        ml_hint = (
            f"Auxiliary signal (do not treat as ground truth): ML classifier suggests "
            f"{lab} (confidence {float(state.get('ml_confidence') or 0):.3f}).\n"
        )

    schema = (
        '{"supported": [], "contradicted": [], "unknown": []}\n'
        "Each array holds short English strings (specific claims or observations). "
        f"At most {_MAX_ITEMS_PER_BUCKET} items per array; each string at most {_MAX_STRING_LEN} characters."
    )

    prompt = (
        "You verify whether a NEWS ARTICLE is consistent with EVIDENCE SNIPPETS from an external corpus.\n"
        "Evidence may be partial, noisy, or only loosely related — use \"unknown\" when you cannot decide.\n\n"
        f"{ml_hint}"
        "Output rules:\n"
        "- Return ONE JSON object only. No markdown fences. No text before or after the JSON.\n"
        f"- Keys must be exactly: {_JSON_KEYS[0]!r}, {_JSON_KEYS[1]!r}, {_JSON_KEYS[2]!r}.\n"
        "- Each value must be a JSON array of strings (use [] if none).\n"
        "- Do not nest objects inside the arrays.\n\n"
        f"Exact shape example (empty): {schema}\n\n"
        "ARTICLE:\n"
        f"{excerpt}\n\n"
        "EVIDENCE:\n"
        f"{evidence}\n"
    )

    try:
        from src.agent.llm_service import generate

        raw = generate(prompt, temperature=0.0, max_tokens=1024)
        buckets = _parse_llm_verification_json(raw)
    except Exception as exc:
        return {
            "verification": _empty_structured(
                mode="fallback",
                llm=False,
                unknown_note=f"verification_unavailable: {exc}",
                chunks_reviewed=len(chunks),
                top_scores=top_scores,
                rag_error=rag_err,
                llm_error=str(exc),
            )
        }

    # Deterministic cleanup: if model returned all empty but we have evidence, one note.
    if not any(buckets[k] for k in _JSON_KEYS):
        buckets["unknown"] = [
            "Model returned no categorized claims; evidence may be weak or unrelated."
        ]

    verification: Dict[str, Any] = {
        "supported": buckets["supported"],
        "contradicted": buckets["contradicted"],
        "unknown": buckets["unknown"],
        "mode": "structured",
        "llm": True,
        "chunks_reviewed": len(chunks),
        "top_scores": top_scores,
    }
    if rag_err:
        verification["rag_error"] = rag_err
    return {"verification": verification}
