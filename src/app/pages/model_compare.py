"""Model Comparison — ROC, PR, confusion matrix, metric bars, CV boxplot, feature importance. Plotly only."""

import streamlit as st

from src.app.components.ui import page_header
from src.app.core import get_expected_metrics, load_model
from src.evaluation.results_loader import get_confusion_matrices
from src.evaluation.plotly_viz import (
    get_lr_feature_importance,
    plotly_confusion_heatmap_from_matrix,
    plotly_confusion_heatmap_reference,
    plotly_cv_boxplot_reference,
    plotly_metric_comparison_bar,
    plotly_pr_reference,
    plotly_roc_reference,
    plotly_top_tfidf_features,
)


def _inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,300&display=swap');

        .stApp { background: #ffffff; }

        .mc-eyebrow {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 6px;
        }
        .mc-heading {
            font-family: 'Fraunces', serif;
            font-size: 22px;
            font-weight: 300;
            color: #0f172a;
            margin: 0 0 20px 0;
            line-height: 1.3;
        }
        .mc-rule {
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 36px 0;
        }

        /* ── Model selector radio ── */
        .mc-selector-label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 10px;
        }
        div[data-testid="stRadio"] label {
            font-family: 'DM Mono', monospace !important;
            font-size: 12px !important;
            color: #475569 !important;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
            border-color: #cbd5e1 !important;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] > div:first-child {
            border-color: #0f172a !important;
            background: #0f172a !important;
        }

        /* ── Chart cards ── */
        .chart-card {
            background: #fafafa;
            border: 1px solid #e8edf2;
            border-radius: 12px;
            padding: 20px 20px 8px;
            margin-bottom: 4px;
        }
        .chart-card-title {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .chart-card-title .dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            background: #0f172a;
            display: inline-block;
            flex-shrink: 0;
        }

        /* ── Notes card ── */
        .notes-card {
            background: #fafafa;
            border: 1px solid #e8edf2;
            border-left: 3px solid #0f172a;
            border-radius: 0 10px 10px 0;
            padding: 20px 24px;
            font-family: 'DM Mono', monospace;
            font-size: 12.5px;
            font-weight: 300;
            color: #475569;
            line-height: 1.75;
        }
        .notes-card strong {
            color: #0f172a;
            font-weight: 500;
        }

        /* ── Feature importance cards ── */
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
        .dot-fake { width:6px;height:6px;border-radius:50%;background:#f87171;display:inline-block;flex-shrink:0; }
        .dot-real { width:6px;height:6px;border-radius:50%;background:#4ade80;display:inline-block;flex-shrink:0; }

        /* ── Info / warn notices ── */
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
        "Model comparison",
        "Compare all trained models on the same TF-IDF features and stratified split.",
    )

    metrics = get_expected_metrics()
    if not metrics:
        st.markdown(
            '<div class="warn-box">⚠ Evaluation results not found. Run the notebook or '
            '<code>python scripts/run_evaluation.py</code> to generate metrics.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Model selector ──
    st.markdown('<p class="mc-eyebrow">01 / Select model</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="mc-heading">Choose a model to inspect</h2>', unsafe_allow_html=True)

    model_names = list(metrics.keys())
    selected = st.radio("Show curves for", model_names, horizontal=True, key="model_select", label_visibility="collapsed")

    m = metrics.get(selected)
    if not m:
        st.stop()

    st.markdown('<hr class="mc-rule">', unsafe_allow_html=True)

    # ── Row 1: ROC + PR ──
    st.markdown('<p class="mc-eyebrow">02 / Curves</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="mc-heading">ROC &amp; Precision-Recall</h2>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="chart-card"><div class="chart-card-title"><span class="dot"></span>ROC Curve</div>', unsafe_allow_html=True)
        fig_roc = plotly_roc_reference(selected, m["ROC-AUC"])
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-card"><div class="chart-card-title"><span class="dot"></span>Precision-Recall Curve</div>', unsafe_allow_html=True)
        fig_pr = plotly_pr_reference(selected, m["Precision"], m["Recall"], m["ROC-AUC"])
        st.plotly_chart(fig_pr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="mc-rule">', unsafe_allow_html=True)

    # ── Row 2: Confusion matrix + Metric bar ──
    st.markdown('<p class="mc-eyebrow">03 / Performance</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="mc-heading">Confusion matrix &amp; metric comparison</h2>', unsafe_allow_html=True)

    c3, c4 = st.columns(2, gap="medium")
    confusion_matrices = get_confusion_matrices()
    with c3:
        st.markdown('<div class="chart-card"><div class="chart-card-title"><span class="dot"></span>Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrices.get(selected) if confusion_matrices else None
        if cm and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
            fig_cm = plotly_confusion_heatmap_from_matrix(cm, selected)
        else:
            fig_cm = plotly_confusion_heatmap_reference(m["Accuracy"], m["Precision"], m["Recall"], selected)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="chart-card"><div class="chart-card-title"><span class="dot"></span>Metric comparison — all models</div>', unsafe_allow_html=True)
        fig_bar = plotly_metric_comparison_bar(metrics)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="mc-rule">', unsafe_allow_html=True)

    # ── CV boxplot ──
    st.markdown('<p class="mc-eyebrow">04 / Stability</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="mc-heading">Cross-validation F1 distribution</h2>', unsafe_allow_html=True)

    has_cv_data = any(
        isinstance(mv.get("CV F1"), (list, tuple)) and len(mv.get("CV F1", [])) >= 2
        for mv in (metrics or {}).values()
    )
    if has_cv_data:
        st.markdown('<div class="chart-card"><div class="chart-card-title"><span class="dot"></span>5-Fold CV F1 boxplot</div>', unsafe_allow_html=True)
        fig_cv = plotly_cv_boxplot_reference(metrics)
        st.plotly_chart(fig_cv, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="info-notice"><span style="font-size:14px;flex-shrink:0">◎</span>'
            '<span>CV F1 not in evaluation results. Run the full notebook or '
            '<code>python scripts/run_evaluation.py</code> to compute 5-fold CV F1, then restart the app.</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="mc-rule">', unsafe_allow_html=True)

    # ── Model notes ──
    st.markdown('<p class="mc-eyebrow">05 / Notes</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="mc-heading">Model notes</h2>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="notes-card">
            TF-IDF produces high-dimensional sparse features.
            <strong>Logistic Regression</strong> often performs well with a linear boundary and L2 regularisation.
            <strong>Naive Bayes</strong> is fast and interpretable;
            <strong>Random Forest</strong> and <strong>SVM</strong> offer alternative decision boundaries.
            The <strong>best model by F1</strong> is saved and used in the Live Prediction Lab.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="mc-rule">', unsafe_allow_html=True)

    # ── Feature importance ──
    st.markdown('<p class="mc-eyebrow">06 / Interpretation</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="mc-heading">Feature importance <span style="font-size:14px;color:#94a3b8;font-family:\'DM Mono\',monospace;font-weight:300;">linear models · LR &amp; SVM</span></h2>', unsafe_allow_html=True)

    try:
        pipeline = load_model()
        names_fake, coefs_fake, names_real, coefs_real = get_lr_feature_importance(pipeline, top_n=15)
        if names_fake and names_real:
            left, right = st.columns(2, gap="medium")
            with left:
                st.markdown('<div class="feat-card"><div class="feat-card-title"><span class="dot-fake"></span>Words → Fake</div>', unsafe_allow_html=True)
                fig_fake = plotly_top_tfidf_features(names_fake, coefs_fake, "Words → Fake", color="#f87171", top_n=15)
                st.plotly_chart(fig_fake, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with right:
                st.markdown('<div class="feat-card"><div class="feat-card-title"><span class="dot-real"></span>Words → Real</div>', unsafe_allow_html=True)
                fig_real = plotly_top_tfidf_features(names_real, coefs_real, "Words → Real", color="#22c55e", top_n=15)
                st.plotly_chart(fig_real, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="info-notice"><span style="font-size:14px;flex-shrink:0">◎</span>'
                '<span>Pipeline has no LR coefficients. Train and add <code>model/pipeline.pkl</code>.</span></div>',
                unsafe_allow_html=True,
            )
    except FileNotFoundError:
        st.markdown(
            '<div class="info-notice"><span style="font-size:14px;flex-shrink:0">◎</span>'
            '<span>Load the trained model (<code>model/pipeline.pkl</code>) to show feature importance.</span></div>',
            unsafe_allow_html=True,
        )