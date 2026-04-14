"""
Deep Analysis — LangGraph agent (ML + RAG + Groq) with a minimal results layout.
"""

from __future__ import annotations

import logging
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
        "Runs the full pipeline on this page: normalize → ML → query planning → "
        "retrieval → verification → report (not the “high-confidence shortcut”). "
        "Groq improves query planning and verification when `GROQ_API_KEY` is set; "
        "build `data/rag/` for non-empty sources.",
    )

    st.markdown('<div class="da-wrap">', unsafe_allow_html=True)
    st.markdown(
        '<p class="da-input-hint">Paste or load a sample, then run. The live pipeline shows each LangGraph node as it executes.</p>',
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
        ok, err = validate_input(text)
        if not ok:
            st.warning(err)
        else:
            try:
                events: list[tuple[str, dict]] = []
                caption = (
                    "Live LangGraph run: each row is one graph node and the source file that executed it."
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
                "unknown items (Groq verification when an API key is set)."
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

    st.markdown("</div>", unsafe_allow_html=True)


def _escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
