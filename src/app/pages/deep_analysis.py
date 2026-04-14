"""
Deep Analysis — LangGraph agent (ML + RAG + LLM) with live node timeline and structured report.
"""

from __future__ import annotations

import logging
import os
import textwrap

import streamlit as st

from src.app.components.agent_pipeline import (
    deep_analysis_results_css,
    pipeline_styles_css,
    timeline_events_html,
)
from src.app.components.ui import page_header
from src.app.core import EXAMPLE_TEXTS, validate_input
from src.agent.graph import iter_credibility_agent
from src.agent.feedback import record_feedback

logger = logging.getLogger("news_credibility_app")

# Any value > 1.0 forces the low-confidence branch so plan → retrieve → verify
# always run (ML probabilities are in [0, 1]). Generic invoke defaults to 0.65.
_DEEP_ANALYSIS_ALWAYS_RAG_THRESHOLD = 1.5


def _inject_styles():
    font_block = textwrap.dedent(
        """
        @import url("https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600&family=Newsreader:ital,opsz,wght@6..72,500;6..72,600&display=swap");
        """
    ).strip()
    st.markdown(
        "<style>"
        + font_block
        + deep_analysis_results_css()
        + pipeline_styles_css()
        + "</style>",
        unsafe_allow_html=True,
    )


def render():
    _inject_styles()

    if "deep_pending_example" in st.session_state:
        st.session_state["deep_input"] = st.session_state.pop("deep_pending_example")
        st.session_state.pop("deep_agent_out", None)
        st.session_state.pop("deep_agent_timeline", None)

    page_header(
        "Deep analysis",
        "Streams **LangGraph** on this page: normalize → ML → (plan → retrieve → verify) → report → "
        "**validate_report**. The UI forces the low-confidence branch so plan/retrieve/verify always run. "
        "Configure **FAISS vs Chroma**, **similarity vs MMR**, and **LLM provider** below. "
        "Gemini falls back to Groq on errors when `GROQ_API_KEY` is set. Build `data/rag/` for non-empty RAG hits.",
    )

    st.warning(
        "**Disclaimer — misinformation & ethics:** This page uses statistical models and "
        "optional LLMs over a **fixed, project-sized** text index. Outputs are **not** legal, "
        "medical, or journalistic advice and **cannot** replace professional fact-checkers or "
        "primary sources. **Do not** treat “High” credibility as proof an article is true; "
        "use it only as a triage signal. Verify consequential claims yourself.",
        icon="⚠️",
    )

    st.session_state.setdefault("rag_backend", "faiss")
    st.session_state.setdefault("rag_search_type", "similarity")
    st.session_state.setdefault("rag_fetch_k", 20)
    st.session_state.setdefault("rag_lambda_mult", 0.6)
    st.session_state.setdefault("llm_provider", "auto")

    with st.expander("Agent runtime — RAG & LLM (this tab)", expanded=True):
        st.caption(
            "These settings apply only to **Deep Analysis** runs. Default RAG is **FAISS** + **similarity**; "
            "Chroma requires `data/rag/chroma_store/` from `scripts/build_chroma_store.py`."
        )
        r1, r2 = st.columns(2)
        with r1:
            st.session_state["rag_backend"] = st.selectbox(
                "RAG backend",
                ["faiss", "chroma"],
                index=0 if st.session_state["rag_backend"] == "faiss" else 1,
                help="FAISS: data/rag/faiss.index + chunks.json. Chroma: data/rag/chroma_store/.",
            )
            st.session_state["rag_search_type"] = st.selectbox(
                "Retrieval mode",
                ["similarity", "mmr"],
                index=0 if st.session_state["rag_search_type"] == "similarity" else 1,
                help="MMR diversifies passages (tune fetch pool and λ below).",
            )
        with r2:
            _lp_opts = ["auto", "groq", "gemini"]
            _lp_cur = str(st.session_state.get("llm_provider") or "auto").strip().lower()
            _lp_idx = _lp_opts.index(_lp_cur) if _lp_cur in _lp_opts else 0
            st.session_state["llm_provider"] = st.selectbox(
                "LLM provider",
                _lp_opts,
                index=_lp_idx,
                help="auto: use secrets/env. gemini: try Gemini, then Groq if Gemini fails and Groq key exists.",
            )
        if st.session_state["rag_search_type"] == "mmr":
            st.session_state["rag_fetch_k"] = int(
                st.slider(
                    "MMR fetch_k",
                    min_value=8,
                    max_value=60,
                    value=int(st.session_state["rag_fetch_k"]),
                    step=2,
                )
            )
            st.session_state["rag_lambda_mult"] = float(
                st.slider(
                    "MMR λ (relevance vs diversity)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state["rag_lambda_mult"]),
                    step=0.05,
                )
            )

    st.markdown('<div class="da-wrap">', unsafe_allow_html=True)
    st.markdown(
        '<p class="da-input-hint">Paste or load a sample, then run. The timeline lists each LangGraph node in order '
        "(including <strong>validate_report</strong>); the highlighted row is the step currently executing.</p>",
        unsafe_allow_html=True,
    )

    text = st.text_area(
        "Article",
        height=200,
        placeholder="Paste article text…",
        key="deep_input",
        label_visibility="collapsed",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Fake sample", key="da_fake"):
            st.session_state["deep_pending_example"] = EXAMPLE_TEXTS["Fake example"]
            st.rerun()
    with c2:
        if st.button("Real sample", key="da_real"):
            st.session_state["deep_pending_example"] = EXAMPLE_TEXTS["Real example"]
            st.rerun()
    with c3:
        if st.button("Clear", key="da_clear"):
            st.session_state["deep_pending_example"] = ""
            st.session_state.pop("deep_agent_out", None)
            st.session_state.pop("deep_agent_timeline", None)
            st.rerun()

    run = st.button("Run deep analysis", type="primary", use_container_width=False)

    if run:
        # Apply Agent runtime (this tab) selections for this run.
        prov = str(st.session_state.get("llm_provider") or "auto").strip().lower()
        if prov == "auto":
            # Let Streamlit secrets / process env control defaults without forcing a value.
            os.environ.pop("LLM_PROVIDER", None)
        elif prov in ("groq", "gemini"):
            os.environ["LLM_PROVIDER"] = prov
        ok, err = validate_input(text)
        if not ok:
            st.warning(err)
        else:
            try:
                events: list[tuple[str, dict]] = []
                caption = (
                    "Live LangGraph stream: each row is one node (normalize → … → validate_report). "
                    "Snippets show RAG backend/mode and LLM_PROVIDER for this run."
                )

                def _tick(node_name: str, merged: dict) -> None:
                    events.append((node_name, dict(merged)))
                    live.markdown(
                        '<p class="da-flow-head">Running · '
                        f"{_escape_html(node_name)}</p>"
                        + timeline_events_html(events, pulse_last=True),
                        unsafe_allow_html=True,
                    )

                live = st.empty()
                if hasattr(st, "status"):
                    with st.status("Credibility agent · running", expanded=True) as run_status:
                        st.caption(caption)
                        for node_name, merged in iter_credibility_agent(
                            text.strip(),
                            confidence_threshold=_DEEP_ANALYSIS_ALWAYS_RAG_THRESHOLD,
                            rag_backend=str(st.session_state.get("rag_backend") or "faiss"),
                            rag_search_type=str(st.session_state.get("rag_search_type") or "similarity"),
                            rag_fetch_k=int(st.session_state.get("rag_fetch_k") or 20),
                            rag_lambda_mult=float(st.session_state.get("rag_lambda_mult") or 0.6),
                        ):
                            _tick(node_name, merged)
                        if events:
                            live.markdown(
                                '<p class="da-flow-head">Pipeline complete</p>'
                                + timeline_events_html(events, pulse_last=False),
                                unsafe_allow_html=True,
                            )
                        run_status.update(
                            label="Credibility agent · finished",
                            state="complete",
                            expanded=False,
                        )
                else:
                    st.caption(caption)
                    for node_name, merged in iter_credibility_agent(
                        text.strip(),
                        confidence_threshold=_DEEP_ANALYSIS_ALWAYS_RAG_THRESHOLD,
                        rag_backend=str(st.session_state.get("rag_backend") or "faiss"),
                        rag_search_type=str(st.session_state.get("rag_search_type") or "similarity"),
                        rag_fetch_k=int(st.session_state.get("rag_fetch_k") or 20),
                        rag_lambda_mult=float(st.session_state.get("rag_lambda_mult") or 0.6),
                    ):
                        _tick(node_name, merged)
                    if events:
                        live.markdown(
                            '<p class="da-flow-head">Pipeline complete</p>'
                            + timeline_events_html(events, pulse_last=False),
                            unsafe_allow_html=True,
                        )

                if not events:
                    st.error("Agent produced no steps. Check logs and dependencies.")
                else:
                    st.session_state["deep_agent_out"] = events[-1][1]
                    st.session_state["deep_agent_timeline"] = events
            except Exception as e:
                logger.exception("Deep analysis failed: %s", e)
                st.error("Analysis failed. Check model files, RAG index, and API keys (see README).")

    out = st.session_state.get("deep_agent_out")
    if not out:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    trace = st.session_state.get("deep_agent_timeline")
    if trace:
        with st.expander("Execution trace (files & steps)", expanded=False):
            st.markdown(
                '<p class="da-flow-head">Last run · node order</p>'
                + timeline_events_html(trace, pulse_last=False),
                unsafe_allow_html=True,
            )

    fr = out.get("final_report") or {}
    verdict = fr.get("verdict") or "Unknown"
    confidence = fr.get("confidence") or ""
    summary = fr.get("summary") or ""
    risks = fr.get("risk_factors") or []
    sources = fr.get("sources") or []
    cred = fr.get("credibility_score") or "Low"
    pattern = fr.get("pattern_detection_summary") or ""
    disc = fr.get("disclaimer") or ""

    # Verdict + confidence (card + semantic tint)
    v_lower = verdict.lower()
    v_slug = v_lower if v_lower in ("fake", "real") else "unknown"
    accent = "#b91c1c" if v_slug == "fake" else "#15803d" if v_slug == "real" else "#64748b"
    verdict_e = _escape_html(str(verdict))
    conf_e = _escape_html(str(confidence))
    st.markdown(
        f'<div class="da-verdict-card da-verdict-{v_slug}">'
        f'<p class="da-verdict" style="color:{accent};">{verdict_e}</p>'
        f'<p class="da-sub">{conf_e}</p></div>',
        unsafe_allow_html=True,
    )

    cred_slug = "high" if str(cred).lower() == "high" else "low"
    cred_e = _escape_html(str(cred))
    st.markdown(
        f'<p class="da-section">Credibility score (rubric)</p>'
        f'<div class="da-cred da-cred-{cred_slug}">'
        f'<span class="da-cred-label">{cred_e}</span>'
        f'<span class="da-cred-hint">High = stronger trust signal from ML + evidence scan; '
        f"Low = fake label, weak confidence, evidence tension, or pipeline issue.</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="da-section">Pattern detection summary</p>', unsafe_allow_html=True)
    if pattern:
        st.markdown(
            f'<div class="da-body">{_escape_html(pattern).replace(chr(10), "<br/>")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="da-body" style="color:#94a3b8;">No pattern summary available.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="da-section">Summary</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="da-body">{_escape_html(summary).replace(chr(10), "<br/>")}</div>', unsafe_allow_html=True)

    st.markdown('<p class="da-section">Risk factors</p>', unsafe_allow_html=True)
    if risks:
        items = "".join(f'<div class="da-li">· {_escape_html(r)}</div>' for r in risks)
        st.markdown(f'<div class="da-body">{items}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="da-body" style="color:#94a3b8;">None flagged.</div>', unsafe_allow_html=True)

    st.markdown('<p class="da-section">Sources</p>', unsafe_allow_html=True)
    if sources:
        for i, s in enumerate(sources, 1):
            ex = _escape_html(str(s.get("excerpt") or ""))
            sc = float(s.get("score") or 0.0)
            st.markdown(
                f'<div class="da-source-card" style="--da-s:{i - 1}"><strong>#{i}</strong> · relevance {sc:.2f}<br/>{ex}'
                f'<div class="da-meta">Local RAG index · not external URLs</div></div>',
                unsafe_allow_html=True,
            )
    else:
        rag_err = (out or {}).get("rag_error")
        if rag_err:
            st.info(str(rag_err))
        else:
            st.markdown(
                '<div class="da-body" style="color:#94a3b8;">No retrieved passages — '
                "retrieval returned no hits for this query.</div>",
                unsafe_allow_html=True,
            )

    with st.expander("Fact checks (structured)", expanded=False):
        rows = fr.get("fact_checks") or []
        if not rows:
            st.caption(
                "No rows yet. After retrieval runs, this lists supported / contradicted / "
                "unknown items (LLM verification when API keys are set — Groq and/or Gemini)."
            )
        for row in rows:
            status = (row.get("status") or "unknown").lower()
            if status not in ("supported", "contradicted", "unknown"):
                status = "unknown"
            tag = (row.get("status") or "unknown").upper()
            finding = _escape_html(str(row.get("finding") or ""))
            st.markdown(
                f'<div class="da-fact" style="--da-s:0">'
                f'<span class="da-fact-tag da-fact-tag-{status}">{_escape_html(tag)}</span>'
                f"<span>{finding}</span></div>",
                unsafe_allow_html=True,
            )

    with st.expander("Feedback (optional)", expanded=False):
        st.caption("Rate the usefulness of this report. Stored locally in `data/feedback/feedback.jsonl`.")
        rating = st.select_slider("Rating", options=[1, 2, 3, 4, 5], value=4)
        notes = st.text_area("Notes (optional)", height=90, placeholder="What was good / missing?")
        if st.button("Submit feedback"):
            try:
                p = record_feedback(
                    raw_text=text or "",
                    verdict=str(verdict),
                    credibility_score=str(cred),
                    rating=int(rating),
                    notes=str(notes or ""),
                    metadata={
                        "page": "deep_analysis",
                        "rag_backend": str(st.session_state.get("rag_backend") or ""),
                        "rag_search_type": str(st.session_state.get("rag_search_type") or ""),
                        "llm_provider": str(st.session_state.get("llm_provider") or ""),
                    },
                )
                st.success(f"Saved feedback to {p}")
            except Exception as exc:
                st.error(f"Feedback save failed: {exc}")

    if disc:
        st.markdown('<p class="da-section">Structured report disclaimer</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="da-disclaimer">{_escape_html(disc)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
