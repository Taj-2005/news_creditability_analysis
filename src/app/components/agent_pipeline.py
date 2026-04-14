"""
Deep Analysis agent pipeline — step labels, file paths, and live timeline HTML.

HTML is emitted on single lines (no leading 4-space lines) so Streamlit markdown
does not treat content as fenced code blocks.
"""

from __future__ import annotations

import html
import textwrap
from typing import Any, Dict, List, Tuple

# Stroke checkmark (24×24) — same visual language as Streamlit status “complete”, not an emoji.
STEP_DONE_CHECK_SVG = (
    '<svg class="da-step-check-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2.25" stroke-linecap="round" stroke-linejoin="round" '
    'focusable="false" aria-hidden="true"><path d="M20 6L9 17l-5-5"/></svg>'
)

# node_id -> (short title, source file, one-line purpose)
NODE_META: Dict[str, Tuple[str, str, str]] = {
    "normalize": (
        "Normalize",
        "src/agent/nodes/normalize.py",
        "Clean and tokenize input for ML and retrieval.",
    ),
    "ml_classify": (
        "ML classify",
        "src/agent/nodes/ml_classify.py",
        "TF-IDF + trained pipeline → Fake / Real scores.",
    ),
    "plan_queries": (
        "Plan queries",
        "src/agent/nodes/plan_queries.py",
        "Groq plans search strings (fallback window if no API key).",
    ),
    "retrieve": (
        "Retrieve",
        "src/agent/nodes/retrieve.py",
        "FAISS + MiniLM over local data/rag index.",
    ),
    "verify": (
        "Verify",
        "src/agent/nodes/verify.py",
        "Groq compares article to chunks → structured JSON.",
    ),
    "report": (
        "Report",
        "src/agent/nodes/report.py",
        "Build final_report for the UI (optional Groq summary).",
    ),
}


def _snippet(node: str, state: Dict[str, Any]) -> str:
    if node == "normalize":
        raw = state.get("cleaned_text") or ""
        t = raw[:120].replace("\n", " ")
        if not t:
            return "Cleaned text ready."
        return f"{len(raw)} chars cleaned · “{t}”…"
    if node == "ml_classify":
        if state.get("error"):
            return f"ML error: {str(state['error'])[:100]}"
        lab = state.get("ml_label")
        conf = state.get("ml_confidence")
        lab_s = "Fake" if lab == 0 else "Real" if lab == 1 else "?"
        return (
            f"Label {lab_s} · predicted-class confidence {float(conf):.1%}"
            if conf is not None
            else "ML scores updated."
        )
    if node == "plan_queries":
        qs = state.get("queries") or []
        err = state.get("llm_query_error")
        base = f"{len(qs)} search line(s) for RAG."
        if err:
            return base + f" Planner note: {str(err)[:140]}…"
        return base
    if node == "retrieve":
        ch = state.get("retrieved_chunks") or []
        rag = state.get("rag_error")
        if rag:
            return f"⚠ {str(rag)[:200]}"
        return f"{len(ch)} passage(s) retrieved from FAISS."
    if node == "verify":
        v = state.get("verification") or {}
        mode = v.get("mode", "")
        nrev = int(v.get("chunks_reviewed") or 0)
        return f"Mode {mode} · reviewed {nrev} chunk(s)."
    if node == "report":
        return "Dashboard payload ready (summary, risks, sources, fact checks)."
    return "—"


def timeline_events_html(
    events: List[Tuple[str, Dict[str, Any]]],
    *,
    pulse_last: bool,
) -> str:
    rows: List[str] = []
    for i, (node, state) in enumerate(events):
        meta = NODE_META.get(node)
        if not meta:
            continue
        title, path, purpose = meta
        snip_raw = _snippet(node, state)
        snip = html.escape(snip_raw).replace("\n", " ")
        title_e = html.escape(title)
        path_e = html.escape(path)
        purpose_e = html.escape(purpose)
        is_last = i == len(events) - 1
        active_cls = " da-flow-row--active" if (pulse_last and is_last) else ""
        done_cls = " da-flow-row--done" if not (pulse_last and is_last) else ""
        num = i + 1
        rows.append(
            f'<div class="da-flow-row{done_cls}{active_cls}" style="--da-i:{i}">'
            f'<div class="da-flow-line" aria-hidden="true"></div>'
            f'<div class="da-flow-badge"><span class="da-badge-num">{num}</span>'
            f'<span class="da-badge-check" aria-hidden="true">{STEP_DONE_CHECK_SVG}</span></div>'
            f'<div class="da-flow-body">'
            f'<div class="da-flow-title">{title_e}</div>'
            f'<div class="da-flow-path">{path_e}</div>'
            f'<div class="da-flow-purpose">{purpose_e}</div>'
            f'<div class="da-flow-snippet">{snip}</div>'
            f"</div></div>"
        )
    inner = "".join(rows)
    return f'<div class="da-flow-shell"><div class="da-flow">{inner}</div></div>'


# --- Live prediction lab (same visual system as Deep Analysis pipeline) ---

LIVE_PREDICTION_STEP_META: Dict[str, Tuple[str, str, str]] = {
    "validate": (
        "Validate input",
        "src/app/core.py",
        "Length, empty text, and max-size checks before inference.",
    ),
    "load_model": (
        "Load model artifact",
        "src/app/core.py",
        "joblib.load — cached with Streamlit for fast repeat calls.",
    ),
    "predict": (
        "Run prediction",
        "src/app/core.py",
        "clean_text() → TF-IDF → classifier → Fake / Real probabilities.",
    ),
}


def _live_prediction_snippet(step: str, detail: Dict[str, Any]) -> str:
    if step == "validate":
        if not detail.get("ok"):
            return str(detail.get("err") or "Validation failed.")
        n = int(detail.get("chars") or 0)
        return f"{n} characters after trim · input valid for inference."
    if step == "load_model":
        if not detail.get("ok"):
            return str(detail.get("err") or "Could not load model.")
        p = detail.get("path_hint") or "model/pipeline.pkl"
        return f"Pipeline ready ({p})."
    if step == "predict":
        if not detail.get("ok"):
            return str(detail.get("err") or "Prediction failed.")
        pred = detail.get("prediction")
        fp = detail.get("fake_prob")
        rp = detail.get("real_prob")
        if pred is None or fp is None or rp is None:
            return "Scores computed."
        lab = "Fake" if int(pred) == 0 else "Real"
        return f"Argmax label: {lab} · P(Fake)={float(fp):.1%} · P(Real)={float(rp):.1%}"
    return "—"


def live_prediction_flow_html(
    events: List[Tuple[str, Dict[str, Any]]],
    *,
    pulse_last: bool,
) -> str:
    """Same shell/row markup as ``timeline_events_html`` for the ML-only live path."""
    rows: List[str] = []
    for i, (step, detail) in enumerate(events):
        meta = LIVE_PREDICTION_STEP_META.get(step)
        if not meta:
            continue
        title, path, purpose = meta
        snip_raw = _live_prediction_snippet(step, detail)
        snip = html.escape(snip_raw).replace("\n", " ")
        title_e = html.escape(title)
        path_e = html.escape(path)
        purpose_e = html.escape(purpose)
        is_last = i == len(events) - 1
        active_cls = " da-flow-row--active" if (pulse_last and is_last) else ""
        done_cls = " da-flow-row--done" if not (pulse_last and is_last) else ""
        warn_cls = " da-flow-row--warn" if detail.get("warn") else ""
        num = i + 1
        rows.append(
            f'<div class="da-flow-row{done_cls}{active_cls}{warn_cls}" style="--da-i:{i}">'
            f'<div class="da-flow-line" aria-hidden="true"></div>'
            f'<div class="da-flow-badge"><span class="da-badge-num">{num}</span>'
            f'<span class="da-badge-check" aria-hidden="true">{STEP_DONE_CHECK_SVG}</span></div>'
            f'<div class="da-flow-body">'
            f'<div class="da-flow-title">{title_e}</div>'
            f'<div class="da-flow-path">{path_e}</div>'
            f'<div class="da-flow-purpose">{purpose_e}</div>'
            f'<div class="da-flow-snippet">{snip}</div>'
            f"</div></div>"
        )
    inner = "".join(rows)
    return f'<div class="da-flow-shell"><div class="da-flow">{inner}</div></div>'


def pipeline_styles_css() -> str:
    """Pipeline timeline + shared motion tokens (inject inside parent ``<style>``)."""
    return textwrap.dedent(
        """
        .da-flow-shell {
          border-radius: 16px;
          padding: 1px;
          margin: 0 0 1rem 0;
          background: linear-gradient(135deg, rgba(59,130,246,0.35), rgba(16,185,129,0.2), rgba(99,102,241,0.15));
          box-shadow: 0 4px 24px rgba(15,23,42,0.06), 0 1px 3px rgba(15,23,42,0.04);
        }
        .da-flow {
          font-family: 'DM Sans', ui-sans-serif, system-ui, sans-serif;
          border-radius: 15px;
          background: linear-gradient(165deg, #ffffff 0%, #f8fafc 55%, #f1f5f9 100%);
          padding: 0.85rem 1rem 1rem 0.65rem;
          border: 1px solid rgba(226,232,240,0.9);
        }
        .da-flow-row {
          display: grid;
          grid-template-columns: 4px 2.25rem 1fr;
          gap: 0 0.75rem;
          align-items: start;
          padding: 0.55rem 0 0.65rem 0.35rem;
          border-bottom: 1px solid rgba(241,245,249,0.95);
          opacity: 0;
          transform: translateY(10px);
          animation: da-row-in 0.55s cubic-bezier(0.22, 1, 0.36, 1) forwards;
          animation-delay: calc(var(--da-i, 0) * 55ms);
          transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.25s ease;
        }
        .da-flow-row:last-child { border-bottom: none; }
        .da-flow-row:hover {
          box-shadow: 0 6px 20px rgba(15,23,42,0.05);
          border-radius: 10px;
        }
        .da-flow-line {
          width: 4px;
          min-height: 2.75rem;
          border-radius: 4px;
          background: linear-gradient(180deg, #e2e8f0, #cbd5e1);
          margin-top: 0.2rem;
          align-self: stretch;
          transition: background 0.35s ease, box-shadow 0.35s ease;
        }
        .da-flow-row--active .da-flow-line {
          background: linear-gradient(180deg, #3b82f6, #6366f1);
          box-shadow: 0 0 12px rgba(59,130,246,0.45);
          animation: da-line-pulse 1.2s ease-in-out infinite;
        }
        .da-flow-row--done .da-flow-line {
          background: linear-gradient(180deg, #6ee7b7, #34d399);
        }
        .da-flow-row--active {
          background: linear-gradient(90deg, rgba(239,246,255,0.95) 0%, rgba(255,255,255,0) 72%);
          border-radius: 12px;
          margin-left: -0.2rem;
          margin-right: -0.2rem;
          padding-left: 0.55rem;
          padding-right: 0.35rem;
          box-shadow: inset 0 0 0 1px rgba(191,219,254,0.6);
          position: relative;
          overflow: hidden;
        }
        .da-flow-row--active::after {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(100deg, transparent 40%, rgba(255,255,255,0.55) 50%, transparent 60%);
          background-size: 200% 100%;
          animation: da-shimmer 2s ease-in-out infinite;
          pointer-events: none;
        }
        .da-flow-badge {
          position: relative;
          z-index: 1;
          width: 2.25rem;
          height: 2.25rem;
          border-radius: 10px;
          border: 1px solid #e2e8f0;
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: ui-monospace, monospace;
          font-size: 0.72rem;
          font-weight: 600;
          color: #64748b;
          background: #f8fafc;
          transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), border-color 0.25s, background 0.25s, color 0.25s;
        }
        .da-badge-check {
          display: none;
          align-items: center;
          justify-content: center;
          line-height: 0;
          animation: da-check-pop 0.45s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        }
        .da-step-check-svg {
          width: 13px;
          height: 13px;
          display: block;
          flex-shrink: 0;
        }
        .da-flow-row--done .da-badge-num { display: none; }
        .da-flow-row--done .da-badge-check { display: flex; }
        .da-flow-row--done .da-flow-badge {
          background: linear-gradient(145deg, #ecfdf5, #d1fae5);
          color: #047857;
          border-color: #a7f3d0;
        }
        .da-flow-row--active .da-flow-badge {
          background: linear-gradient(145deg, #eff6ff, #dbeafe);
          color: #1d4ed8;
          border-color: #93c5fd;
          box-shadow: 0 0 0 3px rgba(59,130,246,0.18);
          animation: da-badge-float 1.25s ease-in-out infinite;
        }
        .da-flow-body { min-width: 0; position: relative; z-index: 1; }
        .da-flow-title {
          font-family: Georgia, 'Times New Roman', serif;
          font-weight: 600;
          font-size: 0.98rem;
          color: #0f172a;
          letter-spacing: -0.02em;
        }
        .da-flow-path {
          font-family: ui-monospace, monospace;
          font-size: 0.64rem;
          color: #64748b;
          margin-top: 0.12rem;
          word-break: break-all;
          opacity: 0.92;
        }
        .da-flow-purpose {
          font-size: 0.72rem;
          color: #94a3b8;
          margin-top: 0.18rem;
          line-height: 1.38;
        }
        .da-flow-snippet {
          font-size: 0.78rem;
          color: #475569;
          margin-top: 0.32rem;
          line-height: 1.48;
          border-left: 3px solid #e2e8f0;
          padding: 0.35rem 0 0.35rem 0.55rem;
          border-radius: 0 8px 8px 0;
          background: rgba(248,250,252,0.65);
        }
        .da-flow-row--done .da-flow-snippet {
          border-left-color: #6ee7b7;
          background: rgba(236,253,245,0.35);
        }
        .da-flow-row--active .da-flow-snippet {
          border-left-color: #60a5fa;
          background: rgba(239,246,255,0.5);
        }
        .da-flow-head {
          font-family: ui-monospace, monospace;
          font-size: 0.62rem;
          font-weight: 600;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          color: #64748b;
          margin: 0 0 0.5rem 0.15rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        .da-flow-head::before {
          content: "";
          flex: 0 0 8px;
          height: 8px;
          border-radius: 50%;
          background: #94a3b8;
          animation: da-head-dot 1s ease-in-out infinite;
        }
        @keyframes da-row-in {
          from { opacity: 0; transform: translateY(12px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes da-shimmer {
          0% { background-position: 100% 0; }
          100% { background-position: -100% 0; }
        }
        @keyframes da-line-pulse {
          0%, 100% { opacity: 1; filter: brightness(1); }
          50% { opacity: 0.85; filter: brightness(1.15); }
        }
        @keyframes da-badge-float {
          0%, 100% { transform: translateY(0) scale(1); }
          50% { transform: translateY(-2px) scale(1.02); }
        }
        @keyframes da-check-pop {
          from { transform: scale(0.2); opacity: 0; }
          to { transform: scale(1); opacity: 1; }
        }
        @keyframes da-head-dot {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.45; transform: scale(0.85); }
        }
        .da-flow-row--warn .da-flow-line {
          background: linear-gradient(180deg, #fecaca, #fca5a5);
        }
        .da-flow-row--warn .da-flow-badge {
          background: linear-gradient(145deg, #fef2f2, #fee2e2);
          color: #b91c1c;
          border-color: #fecaca;
        }
        .da-flow-row--warn.da-flow-row--done .da-flow-badge {
          background: linear-gradient(145deg, #fef2f2, #fee2e2);
          color: #b91c1c;
          border-color: #fecaca;
        }
        .da-flow-row--warn .da-flow-snippet {
          border-left-color: #f87171;
          background: rgba(254,242,242,0.55);
          color: #991b1b;
        }
        .da-flow-row--warn.da-flow-row--active::after {
          animation: none;
        }
        @media (prefers-reduced-motion: reduce) {
          .da-flow-row,
          .da-flow-row--active::after,
          .da-flow-row--active .da-flow-line,
          .da-flow-row--active .da-flow-badge,
          .da-badge-check,
          .da-flow-head::before {
            animation: none !important;
            transition: none !important;
          }
          .da-flow-row {
            opacity: 1 !important;
            transform: none !important;
            animation-delay: 0s !important;
          }
          .da-flow-row:hover { transform: none; }
        }
        """
    ).strip()


def deep_analysis_results_css() -> str:
    """Verdict card, sections, and source list (dedented for ``<style>``)."""
    return textwrap.dedent(
        """
        .da-wrap { max-width: 760px; margin: 0 auto; }
        .da-input-hint {
          font-size: 0.72rem;
          color: #94a3b8;
          margin: -0.25rem 0 0.75rem 0.05rem;
          letter-spacing: 0.02em;
        }
        .da-verdict-card {
          border-radius: 16px;
          padding: 1.15rem 1.25rem 1.2rem;
          margin: 0.5rem 0 1.25rem 0;
          border: 1px solid #e2e8f0;
          background: linear-gradient(145deg, #ffffff, #f8fafc);
          box-shadow: 0 8px 30px rgba(15,23,42,0.06);
          animation: da-card-rise 0.65s cubic-bezier(0.22, 1, 0.36, 1) both;
          animation-delay: 80ms;
        }
        .da-verdict-card.da-verdict-fake {
          border-color: rgba(248,113,113,0.35);
          background: linear-gradient(145deg, #fffefe, #fff1f2);
          box-shadow: 0 8px 28px rgba(185,28,28,0.08);
        }
        .da-verdict-card.da-verdict-real {
          border-color: rgba(52,211,153,0.4);
          background: linear-gradient(145deg, #fafffe, #ecfdf5);
          box-shadow: 0 8px 28px rgba(21,128,61,0.07);
        }
        .da-verdict {
          font-family: 'Newsreader', Georgia, serif;
          font-size: 1.85rem;
          font-weight: 600;
          margin: 0 0 0.35rem 0;
          letter-spacing: -0.03em;
          line-height: 1.15;
        }
        .da-sub {
          font-family: 'DM Sans', ui-sans-serif, system-ui, sans-serif;
          font-size: 0.84rem;
          color: #64748b;
          margin: 0;
          line-height: 1.45;
        }
        .da-section {
          font-family: 'DM Sans', ui-sans-serif, system-ui, sans-serif;
          font-size: 0.65rem;
          font-weight: 700;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          color: #94a3b8;
          margin: 1.35rem 0 0.45rem 0;
          padding-bottom: 0.25rem;
          border-bottom: 1px solid #f1f5f9;
        }
        .da-body {
          font-family: 'DM Sans', ui-sans-serif, system-ui, sans-serif;
          font-size: 0.94rem;
          line-height: 1.68;
          color: #334155;
          animation: da-fade-in 0.5s ease both;
          animation-delay: 0.12s;
        }
        .da-li { margin: 0.4rem 0; padding-left: 0.15rem; }
        .da-source-card {
          font-size: 0.82rem;
          color: #475569;
          border: 1px solid #e2e8f0;
          border-left: 4px solid #94a3b8;
          padding: 0.65rem 0.85rem 0.7rem 0.85rem;
          margin: 0.45rem 0;
          background: #ffffff;
          border-radius: 4px 12px 12px 4px;
          box-shadow: 0 2px 8px rgba(15,23,42,0.04);
          transition: transform 0.22s ease, box-shadow 0.22s ease, border-left-color 0.22s ease;
          animation: da-card-rise 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
          animation-delay: calc(70ms + var(--da-s, 0) * 55ms);
        }
        .da-source-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 10px 28px rgba(15,23,42,0.08);
          border-left-color: #3b82f6;
        }
        .da-meta { font-size: 0.68rem; color: #94a3b8; margin-top: 0.35rem; }
        @keyframes da-card-rise {
          from { opacity: 0; transform: translateY(14px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes da-fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .da-fact {
          display: flex;
          gap: 0.5rem;
          align-items: flex-start;
          padding: 0.45rem 0;
          border-bottom: 1px solid #f1f5f9;
          font-family: 'DM Sans', ui-sans-serif, sans-serif;
          font-size: 0.82rem;
          line-height: 1.45;
          color: #334155;
          animation: da-fade-in 0.4s ease both;
        }
        .da-fact:last-child { border-bottom: none; }
        .da-fact-tag {
          flex: 0 0 auto;
          font-size: 0.62rem;
          font-weight: 700;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          padding: 0.2rem 0.45rem;
          border-radius: 6px;
          margin-top: 0.08rem;
        }
        .da-fact-tag-supported { background: #ecfdf5; color: #047857; }
        .da-fact-tag-contradicted { background: #fef2f2; color: #b91c1c; }
        .da-fact-tag-unknown { background: #f8fafc; color: #64748b; }
        @media (prefers-reduced-motion: reduce) {
          .da-verdict-card, .da-body, .da-source-card {
            animation: none !important;
            transition: none !important;
          }
          .da-source-card:hover { transform: none; }
          .da-fact { animation: none !important; }
        }
        """
    ).strip()
