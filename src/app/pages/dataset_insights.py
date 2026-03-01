"""Dataset Intelligence — class distribution, text length, top TF-IDF. Plotly-only charts."""

import numpy as np
import streamlit as st

from src.app.components.ui import page_header
from src.app.core import DATASET_NAME, get_dataset_size_str, load_model
from src.evaluation.plotly_viz import (
    get_lr_feature_importance,
    plotly_donut_class_distribution,
    plotly_histogram_text_length,
    plotly_top_tfidf_features,
)
from src.evaluation.results_loader import get_dataset_stats


def _inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,300&display=swap');

        .stApp { background: #ffffff; }

        .di-eyebrow {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 6px;
        }
        .di-heading {
            font-family: 'Fraunces', serif;
            font-size: 22px;
            font-weight: 300;
            color: #0f172a;
            margin: 0 0 20px 0;
            line-height: 1.3;
        }
        .di-rule {
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 36px 0;
        }

        /* ── Stat cards ── */
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-bottom: 8px;
        }
        .stat-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px 22px 18px;
            position: relative;
            overflow: hidden;
        }
        .stat-card::after {
            content: '';
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 2px;
            border-radius: 0 0 12px 12px;
        }
        .stat-card.neutral::after { background: linear-gradient(90deg, #94a3b8 0%, transparent 100%); }
        .stat-card.fake::after    { background: linear-gradient(90deg, #f87171 0%, transparent 100%); }
        .stat-card.real::after    { background: linear-gradient(90deg, #4ade80 0%, transparent 100%); }

        .stat-card-label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 10px;
        }
        .stat-card-value {
            font-family: 'Fraunces', serif;
            font-size: 32px;
            font-weight: 300;
            color: #0f172a;
            line-height: 1;
        }
        .stat-card-sub {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            color: #94a3b8;
            margin-top: 6px;
            font-weight: 300;
        }

        /* ── Chart cards ── */
        .chart-card {
            background: #fafafa;
            border: 1px solid #e8edf2;
            border-radius: 12px;
            padding: 20px 20px 8px;
        }
        .chart-card-title {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .chart-card-title .dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            background: #0f172a;
            display: inline-block;
        }
        .chart-caption {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            color: #94a3b8;
            margin-top: 2px;
            margin-bottom: 12px;
        }

        /* ── Feature importance header ── */
        .feat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }
        .feat-card {
            background: #fafafa;
            border: 1px solid #e8edf2;
            border-radius: 12px;
            padding: 20px 20px 8px;
        }
        .feat-card-title {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .feat-card-title .dot-fake { width:6px;height:6px;border-radius:50%;background:#f87171;display:inline-block; }
        .feat-card-title .dot-real { width:6px;height:6px;border-radius:50%;background:#4ade80;display:inline-block; }

        /* ── Info / warning notices ── */
        .info-notice {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 14px 18px;
            font-family: 'DM Mono', monospace;
            font-size: 12px;
            color: #475569;
            line-height: 1.6;
        }
        .info-notice .ni { font-size: 14px; flex-shrink: 0; margin-top: 1px; }

        .warn-box {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 10px;
            padding: 14px 18px;
            font-family: 'DM Mono', monospace;
            font-size: 12px;
            color: #92400e;
            line-height: 1.6;
        }

        h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 300 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sample_text_lengths(seed: int = 42) -> tuple:
    """Illustrative text length samples per class (when no dataset is loaded)."""
    rng = np.random.default_rng(seed)
    fake_len = rng.lognormal(5, 1, 500)
    real_len = rng.lognormal(5.2, 1, 500)
    return list(fake_len), list(real_len)


def render():
    _inject_styles()
    dataset_size_str = get_dataset_size_str()
    page_header("Dataset Intelligence", f"Summary for {DATASET_NAME} · {dataset_size_str} articles")

    stats = get_dataset_stats()
    if not stats:
        st.markdown(
            '<div class="warn-box">⚠ Dataset statistics not found. Run the notebook or '
            '<code>python scripts/run_evaluation.py</code> to generate evaluation results.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    total      = stats.get("after_drop_empty") or stats.get("total_samples")
    class_counts = stats.get("class_counts") or {}
    class_pct    = stats.get("class_pct") or {}

    fake_count = int(class_counts.get("Fake") or class_counts.get(0) or 0)
    real_count = int(class_counts.get("Real") or class_counts.get(1) or 0)
    fake_pct   = class_pct.get("Fake") or class_pct.get(0) or (fake_count / total if total else 0)
    real_pct   = class_pct.get("Real") or class_pct.get(1) or (real_count / total if total else 0)

    if total and (fake_count + real_count) > 0 and abs((fake_pct or 0) + (real_pct or 0) - 1.0) > 0.01:
        fake_pct = fake_count / (fake_count + real_count)
        real_pct = real_count / (fake_count + real_count)

    # ── Section 1: Summary stats ──
    st.markdown('<p class="di-eyebrow">01 / Corpus</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="di-heading">Dataset summary</h2>', unsafe_allow_html=True)

    total_str   = f"{total:,}" if total is not None else "—"
    fake_pct_str = f"{fake_pct:.1%}" if fake_pct else "—"
    real_pct_str = f"{real_pct:.1%}" if real_pct else "—"
    fake_sub    = f"{fake_count:,} articles" if fake_count else ""
    real_sub    = f"{real_count:,} articles" if real_count else ""

    st.markdown(
        f"""
        <div class="stat-grid">
          <div class="stat-card neutral">
            <div class="stat-card-label">Total samples</div>
            <div class="stat-card-value">{total_str}</div>
            <div class="stat-card-sub">after cleaning</div>
          </div>
          <div class="stat-card fake">
            <div class="stat-card-label">Fake</div>
            <div class="stat-card-value">{fake_pct_str}</div>
            <div class="stat-card-sub">{fake_sub}</div>
          </div>
          <div class="stat-card real">
            <div class="stat-card-label">Real</div>
            <div class="stat-card-value">{real_pct_str}</div>
            <div class="stat-card-sub">{real_sub}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="di-rule">', unsafe_allow_html=True)

    # ── Section 2: Class distribution donut ──
    st.markdown('<p class="di-eyebrow">02 / Balance</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="di-heading">Class distribution</h2>', unsafe_allow_html=True)

    if fake_count > 0 or real_count > 0:
        st.markdown('<div class="chart-card"><div class="chart-card-title"><span class="dot"></span>Fake vs Real split</div>', unsafe_allow_html=True)
        fig_donut = plotly_donut_class_distribution(
            counts=[fake_count, real_count],
            labels=["Fake", "Real"],
            colors=["#f87171", "#4ade80"],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-notice"><span class="ni">◎</span><span>No class counts found in evaluation results.</span></div>', unsafe_allow_html=True)

    st.markdown('<hr class="di-rule">', unsafe_allow_html=True)

    # ── Section 3: Text length histogram ──
    st.markdown('<p class="di-eyebrow">03 / Lengths</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="di-heading">Text length distribution</h2>', unsafe_allow_html=True)

    st.markdown(
        '<div class="chart-card">'
        '<div class="chart-card-title"><span class="dot"></span>Word count per article by class</div>'
        '<div class="chart-caption">Illustrative distribution · actual values from training notebook</div>',
        unsafe_allow_html=True,
    )
    lengths_fake, lengths_real = _sample_text_lengths()
    fig_hist = plotly_histogram_text_length(lengths_fake, lengths_real)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="di-rule">', unsafe_allow_html=True)

    # ── Section 4: TF-IDF feature importance ──
    st.markdown('<p class="di-eyebrow">04 / Features</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="di-heading">Top TF-IDF features</h2>', unsafe_allow_html=True)

    try:
        pipeline = load_model()
        names_fake, coefs_fake, names_real, coefs_real = get_lr_feature_importance(pipeline, top_n=15)
        if names_fake and names_real:
            left, right = st.columns(2, gap="medium")
            with left:
                st.markdown(
                    '<div class="feat-card">'
                    '<div class="feat-card-title"><span class="dot-fake"></span>Words → Fake</div>',
                    unsafe_allow_html=True,
                )
                fig_fake = plotly_top_tfidf_features(
                    names_fake, coefs_fake, "Words → Fake", color="#f87171", top_n=15
                )
                st.plotly_chart(fig_fake, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with right:
                st.markdown(
                    '<div class="feat-card">'
                    '<div class="feat-card-title"><span class="dot-real"></span>Words → Real</div>',
                    unsafe_allow_html=True,
                )
                fig_real = plotly_top_tfidf_features(
                    names_real, coefs_real, "Words → Real", color="#4ade80", top_n=15
                )
                st.plotly_chart(fig_real, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="info-notice"><span class="ni">◎</span>'
                '<span>Pipeline has no LR coefficients. Train and add <code>model/pipeline.pkl</code>.</span></div>',
                unsafe_allow_html=True,
            )
    except FileNotFoundError:
        st.markdown(
            '<div class="info-notice"><span class="ni">◎</span>'
            '<span>Load the trained model (<code>model/pipeline.pkl</code>) to show feature importance here.</span></div>',
            unsafe_allow_html=True,
        )