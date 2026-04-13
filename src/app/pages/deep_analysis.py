"""
Deep Analysis — LangGraph agent (ML + RAG + Groq) with a minimal results layout.
"""

from __future__ import annotations

import logging

import streamlit as st

from src.app.components.ui import page_header
from src.app.core import EXAMPLE_TEXTS, validate_input
from src.agent.graph import invoke_credibility_agent

logger = logging.getLogger("news_credibility_app")

# Any value > 1.0 forces the low-confidence branch so plan → retrieve → verify
# always run (ML probabilities are in [0, 1]). Generic invoke defaults to 0.65.
_DEEP_ANALYSIS_ALWAYS_RAG_THRESHOLD = 1.5


def _inject_styles():
    st.markdown(
        """
        <style>
        .da-wrap { max-width: 720px; margin: 0 auto; }
        .da-verdict {
            font-family: 'Fraunces', Georgia, serif;
            font-size: 1.75rem;
            font-weight: 400;
            color: #0f172a;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
        }
        .da-sub {
            font-family: system-ui, sans-serif;
            font-size: 0.8125rem;
            color: #64748b;
            margin-bottom: 1.75rem;
        }
        .da-section {
            font-family: system-ui, sans-serif;
            font-size: 0.6875rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #94a3b8;
            margin: 1.5rem 0 0.5rem 0;
        }
        .da-body {
            font-family: system-ui, sans-serif;
            font-size: 0.9375rem;
            line-height: 1.65;
            color: #334155;
        }
        .da-li { margin: 0.35rem 0; padding-left: 0.1rem; }
        .da-source {
            font-size: 0.8125rem;
            color: #475569;
            border-left: 2px solid #e2e8f0;
            padding: 0.5rem 0 0.5rem 0.75rem;
            margin: 0.5rem 0;
            background: #fafafa;
            border-radius: 0 6px 6px 0;
        }
        .da-meta { font-size: 0.7rem; color: #94a3b8; margin-top: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render():
    _inject_styles()

    if "deep_pending_example" in st.session_state:
        st.session_state["deep_input"] = st.session_state.pop("deep_pending_example")
        st.session_state.pop("deep_agent_out", None)

    page_header(
        "Deep analysis",
        "Runs the full pipeline on this page: normalize → ML → query planning → "
        "retrieval → verification → report (not the “high-confidence shortcut”). "
        "Groq improves query planning and verification when `GROQ_API_KEY` is set; "
        "build `data/rag/` for non-empty sources.",
    )

    st.markdown('<div class="da-wrap">', unsafe_allow_html=True)

    text = st.text_area(
        "Article",
        height=180,
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
            st.rerun()

    run = st.button("Run deep analysis", type="primary", use_container_width=False)

    if run:
        ok, err = validate_input(text)
        if not ok:
            st.warning(err)
        else:
            try:
                with st.spinner("Running agent…"):
                    out = invoke_credibility_agent(
                        text.strip(),
                        confidence_threshold=_DEEP_ANALYSIS_ALWAYS_RAG_THRESHOLD,
                    )
                st.session_state["deep_agent_out"] = out
            except Exception as e:
                logger.exception("Deep analysis failed: %s", e)
                st.error("Analysis failed. Check model files, RAG index, and API keys (see README).")

    out = st.session_state.get("deep_agent_out")
    if not out:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    fr = out.get("final_report") or {}
    verdict = fr.get("verdict") or "Unknown"
    confidence = fr.get("confidence") or ""
    summary = fr.get("summary") or ""
    risks = fr.get("risk_factors") or []
    sources = fr.get("sources") or []

    # Verdict + confidence
    v_lower = verdict.lower()
    accent = "#b91c1c" if v_lower == "fake" else "#15803d" if v_lower == "real" else "#64748b"
    st.markdown(
        f'<p class="da-verdict" style="color:{accent};">{verdict}</p>'
        f'<p class="da-sub">{confidence}</p>',
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
                f'<div class="da-source"><strong>#{i}</strong> · relevance {sc:.2f}<br/>{ex}'
                f'<div class="da-meta">Local RAG index · not external URLs</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="da-body" style="color:#94a3b8;">No retrieved passages — '
            "usually the FAISS index is missing or retrieval failed. "
            "Build it with: <code>python scripts/build_rag_index.py</code></div>",
            unsafe_allow_html=True,
        )
        rag_err = (out or {}).get("rag_error")
        if rag_err:
            st.caption(str(rag_err))

    with st.expander("Fact checks (structured)", expanded=False):
        rows = fr.get("fact_checks") or []
        if not rows:
            st.caption(
                "No rows yet. After retrieval runs, this lists supported / contradicted / "
                "unknown items (Groq verification when an API key is set)."
            )
        for row in rows:
            st.caption(f"{row.get('status', '').upper()} · {row.get('finding', '')}")

    st.markdown("</div>", unsafe_allow_html=True)


def _escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
