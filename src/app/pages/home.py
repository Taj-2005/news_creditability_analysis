"""Overview — Executive summary, big KPI cards, product pitch. Production-grade layout."""

import streamlit as st

from src.app.components.ui import page_header
from src.app.core import (
    DATASET_NAME,
    get_dataset_size_str,
    get_expected_metrics,
    get_model_algorithm_display,
    PAGE_TITLE,
)


def _inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,300&display=swap');

        .stApp { background: #ffffff; }

        /* ── Eyebrow labels ── */
        .ov-eyebrow {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 6px;
        }

        /* ── Section headings ── */
        .ov-heading {
            font-family: 'Fraunces', serif;
            font-size: 22px;
            font-weight: 300;
            color: #0f172a;
            margin: 0 0 16px 0;
            line-height: 1.3;
        }

        /* ── Divider ── */
        .ov-rule {
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 36px 0;
        }

        /* ── Problem statement block ── */
        .problem-block {
            background: #fafafa;
            border: 1px solid #e8edf2;
            border-left: 3px solid #0f172a;
            border-radius: 0 10px 10px 0;
            padding: 20px 24px;
            font-family: 'DM Mono', monospace;
            font-size: 13px;
            font-weight: 300;
            color: #334155;
            line-height: 1.7;
        }
        .problem-block strong {
            color: #0f172a;
            font-weight: 500;
        }

        /* ── Info pill / notice ── */
        .info-notice {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 14px 18px;
            margin-top: 16px;
            font-family: 'DM Mono', monospace;
            font-size: 12px;
            color: #475569;
            line-height: 1.6;
        }
        .info-notice .notice-icon {
            font-size: 15px;
            flex-shrink: 0;
            margin-top: 1px;
        }
        .info-notice a {
            color: #0f172a;
            text-decoration: underline;
            text-underline-offset: 3px;
        }

        /* ── KPI cards ── */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin: 8px 0 24px;
        }
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px 20px 18px;
            position: relative;
            overflow: hidden;
            transition: border-color 0.2s;
        }
        .kpi-card::after {
            content: '';
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #0f172a 0%, transparent 100%);
            border-radius: 0 0 12px 12px;
        }
        .kpi-label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 10px;
        }
        .kpi-value {
            font-family: 'Fraunces', serif;
            font-size: 34px;
            font-weight: 300;
            color: #0f172a;
            line-height: 1;
            letter-spacing: -0.5px;
        }
        .kpi-sub {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            color: #94a3b8;
            margin-top: 6px;
            font-weight: 300;
        }

        /* ── Metric bars ── */
        .metric-bars {
            display: flex;
            flex-direction: column;
            gap: 14px;
            margin: 4px 0;
        }
        .metric-bar-row {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .metric-bar-label {
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            color: #64748b;
            width: 80px;
            flex-shrink: 0;
        }
        .metric-bar-track {
            flex: 1;
            height: 4px;
            background: #f1f5f9;
            border-radius: 4px;
            overflow: hidden;
        }
        .metric-bar-fill {
            height: 100%;
            background: #0f172a;
            border-radius: 4px;
            transition: width 0.6s ease;
        }
        .metric-bar-pct {
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            color: #0f172a;
            font-weight: 500;
            width: 44px;
            text-align: right;
            flex-shrink: 0;
        }

        /* ── Dataset stat chips ── */
        .dataset-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin: 8px 0 16px;
        }
        .dataset-chip {
            background: #fafafa;
            border: 1px solid #e8edf2;
            border-radius: 10px;
            padding: 16px 20px;
        }
        .dataset-chip-label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 8px;
        }
        .dataset-chip-value {
            font-family: 'Fraunces', serif;
            font-size: 20px;
            font-weight: 300;
            color: #0f172a;
        }

        /* ── Pipeline steps ── */
        .pipeline-steps {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
            margin: 12px 0 20px;
        }
        .pipeline-step {
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 6px 13px;
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            color: #334155;
            white-space: nowrap;
        }
        .pipeline-step.dark {
            background: #0f172a;
            color: #f8fafc;
            border-color: #0f172a;
        }
        .pipeline-arrow {
            color: #cbd5e1;
            font-size: 12px;
        }

        /* ── Product pitch card ── */
        .pitch-card {
            background: #0f172a;
            border-radius: 14px;
            padding: 32px 36px;
            position: relative;
            overflow: hidden;
        }
        .pitch-card::before {
            content: '';
            position: absolute;
            top: -60px; right: -60px;
            width: 200px; height: 200px;
            border-radius: 50%;
            background: rgba(255,255,255,0.03);
        }
        .pitch-card-label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #475569;
            margin-bottom: 12px;
        }
        .pitch-card-body {
            font-family: 'DM Mono', monospace;
            font-size: 13px;
            font-weight: 300;
            color: #94a3b8;
            line-height: 1.75;
        }
        .pitch-card-body strong {
            color: #f1f5f9;
            font-weight: 400;
        }

        /* ── Warning box ── */
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


def render():
    _inject_styles()
    page_header(
        "News Credibility Analyzer",
        "This system detects whether a news article is Fake or Real using machine learning."
    )

    # ── Section 1: What it does ──
    st.markdown('<p class="ov-eyebrow">01 / Problem</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="ov-heading">What it does</h2>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="problem-block">
            Misinformation spreads faster than manual fact-checking can scale.
            This app provides <strong>automated binary classification</strong> of news as
            <strong>Fake</strong> or <strong>Real</strong>, using a reproducible ML pipeline —
            interpretable models, no LLMs, no GPU required.
        </div>
        <div class="info-notice">
            <span class="notice-icon">◎</span>
            <span>Trained on the
            <a href="https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset" target="_blank">
            Kaggle Fake and Real News Dataset</a>.
            For best results, use <strong>US-based English articles</strong> matching the training distribution
            in style and topic.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="ov-rule">', unsafe_allow_html=True)

    # ── Section 2: KPI metrics ──
    metrics = get_expected_metrics()
    if not metrics:
        st.markdown(
            '<div class="warn-box">⚠ Evaluation results not found. Run the notebook or '
            '<code>python scripts/run_evaluation.py</code> to generate metrics, then restart the app.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    best_name = get_model_algorithm_display()
    lr = metrics.get(best_name) or metrics.get("Logistic Regression") or next(iter(metrics.values()), None)
    if not lr:
        st.markdown('<div class="warn-box">⚠ No model metrics found in evaluation results.</div>', unsafe_allow_html=True)
        st.stop()

    acc  = float(lr["Accuracy"])
    f1   = float(lr["F1 Score"])
    auc  = float(lr["ROC-AUC"])

    # CV F1
    cv_str = "—"
    cv_sub = "run scripts/run_evaluation.py"
    cv = lr.get("CV F1")
    if isinstance(cv, (list, tuple)) and len(cv) >= 2:
        cv_str = f"{float(cv[0]):.1%}"
        cv_sub = f"± {float(cv[1]):.2%} std"
    else:
        for m in (metrics or {}).values():
            if isinstance(m.get("CV F1"), (list, tuple)) and len(m["CV F1"]) >= 2:
                cv_str = f"{float(m['CV F1'][0]):.1%}"
                cv_sub = f"± {float(m['CV F1'][1]):.2%} std"
                break

    st.markdown('<p class="ov-eyebrow">02 / Performance</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="ov-heading">Key performance metrics</h2>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-label">Accuracy</div>
            <div class="kpi-value">{acc:.1%}</div>
            <div class="kpi-sub">Overall correct predictions</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">F1 Score</div>
            <div class="kpi-value">{f1:.1%}</div>
            <div class="kpi-sub">Fake-class F1</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">ROC-AUC</div>
            <div class="kpi-value">{auc:.1%}</div>
            <div class="kpi-sub">Ranking / discrimination</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">5-Fold CV F1</div>
            <div class="kpi-value">{cv_str}</div>
            <div class="kpi-sub">{cv_sub}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric bars
    st.markdown(
        f"""
        <div class="metric-bars">
          <div class="metric-bar-row">
            <span class="metric-bar-label">Accuracy</span>
            <div class="metric-bar-track"><div class="metric-bar-fill" style="width:{acc*100:.1f}%"></div></div>
            <span class="metric-bar-pct">{acc:.1%}</span>
          </div>
          <div class="metric-bar-row">
            <span class="metric-bar-label">F1 Score</span>
            <div class="metric-bar-track"><div class="metric-bar-fill" style="width:{f1*100:.1f}%"></div></div>
            <span class="metric-bar-pct">{f1:.1%}</span>
          </div>
          <div class="metric-bar-row">
            <span class="metric-bar-label">ROC-AUC</span>
            <div class="metric-bar-track"><div class="metric-bar-fill" style="width:{auc*100:.1f}%"></div></div>
            <span class="metric-bar-pct">{auc:.1%}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="ov-rule">', unsafe_allow_html=True)

    # ── Section 3: Dataset summary ──
    st.markdown('<p class="ov-eyebrow">03 / Data</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="ov-heading">Dataset summary</h2>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="dataset-grid">
          <div class="dataset-chip">
            <div class="dataset-chip-label">Dataset</div>
            <div class="dataset-chip-value">{DATASET_NAME}</div>
          </div>
          <div class="dataset-chip">
            <div class="dataset-chip-label">Articles</div>
            <div class="dataset-chip-value">{get_dataset_size_str()}</div>
          </div>
          <div class="dataset-chip">
            <div class="dataset-chip-label">Task</div>
            <div class="dataset-chip-value">Fake vs Real</div>
          </div>
        </div>
        <div class="info-notice">
          <span class="notice-icon">◎</span>
          <span>Source: <a href="https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset" target="_blank">Fake and Real News (Kaggle)</a>.
          Best results with US-based, English news matching the training distribution.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="ov-rule">', unsafe_allow_html=True)

    # ── Section 4: Product pitch ──
    st.markdown('<p class="ov-eyebrow">04 / Pipeline</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="ov-heading">End-to-end pipeline</h2>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="pipeline-steps">
          <span class="pipeline-step">Raw Text</span>
          <span class="pipeline-arrow">→</span>
          <span class="pipeline-step">TF-IDF</span>
          <span class="pipeline-arrow">→</span>
          <span class="pipeline-step">LR / NB / RF / SVM</span>
          <span class="pipeline-arrow">→</span>
          <span class="pipeline-step">Evaluation</span>
          <span class="pipeline-arrow">→</span>
          <span class="pipeline-step">Streamlit App</span>
          <span class="pipeline-arrow">→</span>
          <span class="pipeline-step dark">Live Verdict</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pitch-card">
          <div class="pitch-card-label">Product pitch</div>
          <div class="pitch-card-body">
            <strong>Problem → Data → Model → Evaluation → Deployment → Live Testing.</strong><br><br>
            This milestone delivers an end-to-end pipeline: from raw text to a credibility verdict,
            with stratified splits, class balancing, and four compared models —
            <strong>Logistic Regression</strong>, <strong>Naive Bayes</strong>,
            <strong>Random Forest</strong>, and <strong>SVM</strong> — surfaced through
            a Streamlit app suitable for demos and further agentic AI integration.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )