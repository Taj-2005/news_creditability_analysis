"""
Milestone 2 agentic runtime — animated pipeline HTML/CSS for the Architecture page.

Uses a wrapper class switch (ML-only vs full agent) so Streamlit rerenders one
``st.markdown`` block; no external JS. Single-line HTML segments avoid Markdown
code-fence parsing issues.
"""

from __future__ import annotations

import html
import textwrap
from typing import Literal

ModeKey = Literal["ml_only", "agent"]


def architecture_pipeline_css() -> str:
    return textwrap.dedent(
        """
        .arch-pipe-wrap{font-family:ui-sans-serif,system-ui,sans-serif;max-width:820px;margin:0 auto 2rem;padding:0 4px;}
        .arch-pipe{font-size:13px;color:#334155;}
        .arch-legend{display:flex;flex-wrap:wrap;gap:10px 16px;margin:0 0 18px;font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:#64748b;}
        .arch-legend span{display:inline-flex;align-items:center;gap:6px;}
        .arch-legend i{width:8px;height:8px;border-radius:2px;}
        .lg-ml{background:#2563eb;}
        .lg-rag{background:#7c3aed;}
        .lg-llm{background:#059669;}
        .lg-out{background:#ea580c;}
        .arch-col{display:flex;flex-direction:column;align-items:center;}
        .arch-conn{width:2px;height:22px;margin:0 auto;position:relative;overflow:hidden;border-radius:2px;background:#e2e8f0;}
        .arch-conn::after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,transparent,#94a3b8,transparent);animation:arch-flow-dash 1.8s linear infinite;opacity:.85;}
        @keyframes arch-flow-dash{0%{transform:translateY(-100%);}100%{transform:translateY(100%);}}
        .arch-node{position:relative;width:100%;max-width:420px;border-radius:14px;padding:14px 16px 14px 18px;margin:0 auto;border:1px solid #e2e8f0;background:linear-gradient(165deg,#fff,#f8fafc);box-shadow:0 2px 12px rgba(15,23,42,.04);transition:opacity .35s ease,transform .35s ease,box-shadow .35s ease,border-color .35s;}
        .arch-node-glow{position:absolute;inset:-1px;border-radius:14px;opacity:0;pointer-events:none;transition:opacity .4s;}
        .arch-node-k{font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:4px;}
        .arch-node-t{font-size:1.05rem;font-weight:700;color:#0f172a;letter-spacing:-.02em;}
        .arch-node-s{font-size:11px;color:#64748b;margin-top:6px;line-height:1.45;font-family:ui-monospace,monospace;}
        .arch-node--ml{border-left:4px solid #2563eb;}
        .arch-node--ml .arch-node-k{color:#2563eb;}
        .arch-node--rag{border-left:4px solid #7c3aed;}
        .arch-node--rag .arch-node-k{color:#7c3aed;}
        .arch-node--llm{border-left:4px solid #059669;}
        .arch-node--llm .arch-node-k{color:#059669;}
        .arch-node--out{border-left:4px solid #ea580c;}
        .arch-node--out .arch-node-k{color:#ea580c;}
        .arch-node--neutral{border-left:4px solid #64748b;}
        .arch-node--val{border-left:4px solid #d97706;}
        .arch-node--val .arch-node-k{color:#d97706;}
        .arch-node--router{text-align:center;max-width:340px;padding:12px 16px;border-radius:999px;background:#f1f5f9;border:1px dashed #94a3b8;font-weight:600;font-size:12px;color:#334155;}
        .arch-node--dim .arch-node-glow{opacity:0!important;}
        .arch-split{display:flex;width:100%;max-width:640px;gap:20px;margin:8px auto 0;align-items:stretch;justify-content:center;flex-wrap:wrap;}
        .arch-branch{flex:1;min-width:200px;max-width:280px;display:flex;flex-direction:column;align-items:center;}
        .arch-branch-cap{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#94a3b8;margin-bottom:8px;text-align:center;}
        .arch-pipe--ml_only .arch-agent-only{opacity:.28;filter:grayscale(.35);transform:scale(.98);}
        .arch-pipe--ml_only .arch-ml-short{opacity:1;animation:arch-node-pulse 2.2s ease-in-out infinite;}
        .arch-pipe--ml_only .arch-core-path{animation:arch-node-pulse 2.4s ease-in-out infinite;}
        .arch-pipe--agent .arch-ml-short{opacity:.34;filter:grayscale(.25);}
        .arch-pipe--agent .arch-agent-only{animation:arch-agent-glow 2.2s ease-in-out infinite;}
        .arch-pipe--agent .arch-core-path{opacity:1;}
        @keyframes arch-node-pulse{0%,100%{box-shadow:0 2px 12px rgba(15,23,42,.06);}50%{box-shadow:0 4px 22px rgba(37,99,235,.2);}}
        @keyframes arch-agent-glow{0%,100%{box-shadow:0 2px 12px rgba(15,23,42,.06);}33%{box-shadow:0 4px 20px rgba(124,58,237,.18);}66%{box-shadow:0 4px 20px rgba(5,150,105,.2);}}
        @media (prefers-reduced-motion:reduce){.arch-conn::after,.arch-node,.arch-pipe--ml_only .arch-ml-short,.arch-pipe--ml_only .arch-core-path,.arch-pipe--agent .arch-agent-only{animation:none!important;}.arch-conn::after{opacity:0;}}
        .arch-pipe-foot{font-size:11px;color:#64748b;margin-top:14px;line-height:1.5;max-width:640px;text-align:center;}
        """
    ).strip()


def _n(
    *,
    cat: str,
    title: str,
    subtitle: str,
    extra: str = "",
) -> str:
    """One pipeline card; ``cat`` is ml|rag|llm|val|out|neutral."""
    t_esc = html.escape(title)
    s_esc = html.escape(subtitle)
    base = f"arch-node arch-node--{cat}"
    if extra == "agent":
        base += " arch-node--dim arch-agent-only"
    elif extra == "short":
        base += " arch-node--dim arch-ml-short"
    elif extra == "core":
        base += " arch-core-path"
    elif extra == "short_hi":
        base += " arch-ml-short"
    elif extra == "agent_hi":
        base += " arch-agent-only"
    cls = base
    k = cat.upper() if cat != "neutral" else "IO"
    if cat == "ml":
        k = "ML"
    elif cat == "rag":
        k = "RAG"
    elif cat == "llm":
        k = "LLM"
    elif cat == "val":
        k = "CHK"
    elif cat == "out":
        k = "OUT"
    return "".join(
        [
            f'<div class="{cls}">',
            '<div class="arch-node-glow"></div>',
            f'<div class="arch-node-k">{k}</div>',
            f'<div class="arch-node-t">{t_esc}</div>',
            f'<div class="arch-node-s">{s_esc}</div>',
            "</div>",
        ]
    )


def _c() -> str:
    return '<div class="arch-col"><div class="arch-conn"></div></div>'


def _router() -> str:
    return (
        '<div class="arch-col">'
        '<div class="arch-conn"></div>'
        '<div class="arch-node arch-node--router arch-core-path">'
        "ml_confidence &lt; threshold → full lane (plan → RAG → verify) → report → validate · "
        "else → report → validate (skips RAG)"
        "</div>"
        '<div class="arch-conn"></div>'
        "</div>"
    )


def build_pipeline_markup(mode: ModeKey) -> str:
    """Full pipeline HTML; ``mode`` switches CSS emphasis class on wrapper."""
    wrap_cls = "arch-pipe-wrap arch-pipe arch-pipe--ml_only" if mode == "ml_only" else "arch-pipe-wrap arch-pipe arch-pipe--agent"
    legend = "".join(
        [
            '<div class="arch-legend">',
            '<span><i class="lg-ml"></i> ML</span>',
            '<span><i class="lg-rag"></i> RAG</span>',
            '<span><i class="lg-llm"></i> LLM</span>',
            '<span><i class="lg-out"></i> Output</span>',
            "</div>",
        ]
    )
    parts: list[str] = [
        f'<div class="{wrap_cls}">',
        legend,
        '<div class="arch-col">',
        _n(
            cat="neutral",
            title="User input",
            subtitle="Raw article / headline · Streamlit Deep Analysis or API-style invoke",
            extra="core",
        ),
        "</div>",
        _c(),
        '<div class="arch-col">',
        _n(
            cat="ml",
            title="Preprocess",
            subtitle="normalize.py · clean_text(), same contract as training",
            extra="core",
        ),
        "</div>",
        _c(),
        '<div class="arch-col">',
        _n(
            cat="ml",
            title="ML classifier",
            subtitle="TF-IDF + SVM/LR/etc. · ml_classify.py · scores + calibrated path routing",
            extra="core",
        ),
        "</div>",
        _c(),
        _router(),
        '<div class="arch-split">',
        '<div class="arch-branch">',
        '<div class="arch-branch-cap">Higher ML confidence · report without RAG</div>',
        _c(),
        _n(
            cat="out",
            title="Structured report",
            subtitle="report.py + ui_report.py · skips plan/retrieve/verify on this lane · LLM optional for narrative",
            extra="short_hi" if mode == "ml_only" else "short",
        ),
        "</div>",
        '<div class="arch-branch">',
        '<div class="arch-branch-cap">Lower confidence · plan → retrieve → verify</div>',
        _c(),
        _n(
            cat="llm",
            title="Query generation",
            subtitle="plan_queries.py · LLM search strings (Groq/Gemini) · text-window fallback if no key",
            extra="agent_hi" if mode == "agent" else "agent",
        ),
        _c(),
        _n(
            cat="rag",
            title="RAG retrieval",
            subtitle="retrieve.py · FAISS or Chroma · similarity or MMR · configured on Deep Analysis",
            extra="agent_hi" if mode == "agent" else "agent",
        ),
        _c(),
        _n(
            cat="llm",
            title="LLM reasoning & verification",
            subtitle="verify.py · evidence JSON vs chunks (Gemini → Groq fallback when both keys exist)",
            extra="agent_hi" if mode == "agent" else "agent",
        ),
        _c(),
        _n(
            cat="out",
            title="Structured report",
            subtitle="report.py + ui_report.py · merges ML, verification, and RAG context",
            extra="agent_hi" if mode == "agent" else "agent",
        ),
        "</div>",
        "</div>",
        _c(),
        '<div class="arch-col">',
        _n(
            cat="val",
            title="Validate report",
            subtitle="validate_report.py · schema checks · at most one retry of report",
            extra="core",
        ),
        "</div>",
        _c(),
        '<div class="arch-col">',
        _n(
            cat="out",
            title="Final output",
            subtitle="final_report → UI · Live Prediction Lab = ML-only shortcut (no LangGraph)",
            extra="core",
        ),
        "</div>",
        '<p class="arch-pipe-foot">LangGraph (Deep Analysis): normalize → ml_classify → branch → … → report → validate_report → end. '
        "This page’s toggle only changes <strong>emphasis</strong> (ML-only story vs full agent). "
        "RAG backend + MMR + LLM provider are set under <strong>Deep Analysis → Agent runtime</strong>.</p>",
        "</div>",
    ]
    return "".join(parts)


def render_architecture_toggle_caption() -> str:
    """Markdown caption below the path selector."""
    return (
        "**ML fast path** — **Live Prediction Lab**: `clean_text` + loaded `pipeline.pkl` → Fake/Real probabilities (no LangGraph). "
        "**Full agent** — **Deep Analysis**: streamed LangGraph (`iter_credibility_agent`): plan → retrieve (FAISS/Chroma, similarity/MMR) → "
        "verify → report → **validate_report** (bounded retry). Pick RAG + LLM settings on the Deep Analysis page."
    )
