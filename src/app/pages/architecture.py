"""Architecture — system diagram (Mermaid), train vs inference, repo mapping. Enhanced card-based layout."""

import streamlit as st

from src.app.components.architecture_flow import (
    architecture_pipeline_css,
    build_pipeline_markup,
    render_architecture_toggle_caption,
)
from src.app.components.ui import page_header
from src.app.core import get_model_algorithm_display, MODEL_FILENAME, MODEL_DIR_NAME


# ── Inject global styles ──────────────────────────────────────────────────────
def _inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,300&display=swap');

        /* ── Reset & base ── */
        .stApp { background: #ffffff; }

        /* ── Section labels ── */
        .arch-eyebrow {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 6px;
        }

        /* ── Section headings ── */
        .arch-heading {
            font-family: 'Fraunces', serif;
            font-size: 22px;
            font-weight: 300;
            color: #0f172a;
            margin: 0 0 4px 0;
            line-height: 1.3;
        }

        /* ── Divider line ── */
        .arch-rule {
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 32px 0;
        }

        /* ── Cards ── */
        .arch-card {
            background: #fafafa;
            border: 1px solid #e8edf2;
            border-radius: 12px;
            padding: 24px 28px;
            height: 100%;
            position: relative;
            overflow: visible;
        }
        .arch-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #cbd5e1 0%, transparent 100%);
            border-radius: 12px 12px 0 0;
        }
        .arch-card-accent::before {
            background: linear-gradient(90deg, #0f172a 0%, #475569 100%);
        }

        .arch-card-title {
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .arch-card-title .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #0f172a;
            display: inline-block;
        }

        /* ── File/step chips ── */
        .file-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 8px;
        }
        .file-item {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            font-size: 13px;
            line-height: 1.5;
            color: #334155;
            font-family: 'DM Mono', monospace;
            font-weight: 300;
        }
        .file-item .file-name {
            color: #0f172a;
            font-weight: 500;
            white-space: nowrap;
        }
        .file-item .file-desc {
            color: #64748b;
            font-weight: 300;
            font-family: 'DM Mono', monospace;
            font-size: 12px;
        }
        .file-item .arrow {
            color: #cbd5e1;
            font-size: 10px;
            margin-top: 3px;
            flex-shrink: 0;
        }

        /* ── Flow step badges ── */
        .flow-steps {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 10px;
            align-items: center;
        }
        .flow-step {
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 5px 11px;
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            color: #334155;
            white-space: nowrap;
        }
        .flow-arrow {
            color: #cbd5e1;
            font-size: 13px;
            font-family: monospace;
        }
        .flow-step.highlight {
            background: #0f172a;
            color: #f8fafc;
            border-color: #0f172a;
        }

        /* ── Repo table ── */
        .repo-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'DM Mono', monospace;
            font-size: 12.5px;
            margin-top: 4px;
        }
        .repo-table th {
            text-align: left;
            font-size: 10px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #94a3b8;
            font-weight: 500;
            padding: 8px 12px;
            border-bottom: 1px solid #e2e8f0;
        }
        .repo-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #f1f5f9;
            color: #334155;
            vertical-align: top;
        }
        .repo-table td:first-child {
            color: #0f172a;
            font-weight: 500;
            white-space: nowrap;
        }
        .repo-table tr:last-child td { border-bottom: none; }
        .repo-table tr:hover td { background: #f8fafc; }

        /* ── Mermaid container ── */
        .mermaid-wrap {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 16px 8px 8px;
            overflow: visible;
        }
        .mermaid-label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #94a3b8;
            padding: 0 12px 10px;
        }

        /* ── Datasource pill ── */
        .ds-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 4px 14px 4px 8px;
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            color: #475569;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .ds-pill span.icon { font-size: 13px; }

        /* ── Page header tweak ── */
        h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 300 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Mermaid renderer ──────────────────────────────────────────────────────────
def _mermaid_html(diagram: str, height: int = 280, label: str = "") -> str:
    label_html = (
        f'<div style="font-family:monospace;font-size:10px;letter-spacing:.14em;'
        f'text-transform:uppercase;color:#94a3b8;padding:0 12px 10px;">{label}</div>'
        if label else ""
    )
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: #f8fafc;
    font-family: monospace;
  }}
  .wrap {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 8px 12px;
    overflow: visible;

  }}
  .mermaid {{
    text-align: center;
  }}
  /* Mermaid SVG overrides */
  .mermaid svg {{
    max-width: 100%;
    height: auto;
  }}
</style>
</head>
<body>
<div class="wrap">
  {label_html}
  <div class="mermaid">
{diagram}
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({{
    startOnLoad: true,
    theme: 'base',
    themeVariables: {{
      primaryColor: '#f1f5f9',
      primaryTextColor: '#0f172a',
      primaryBorderColor: '#cbd5e1',
      lineColor: '#94a3b8',
      secondaryColor: '#f8fafc',
      tertiaryColor: '#ffffff',
      background: '#f8fafc',
      mainBkg: '#f1f5f9',
      nodeBorder: '#cbd5e1',
      clusterBkg: '#ffffff',
      clusterBorder: '#e2e8f0',
      titleColor: '#0f172a',
      edgeLabelBackground: '#ffffff',
      fontFamily: 'monospace',
      fontSize: '12px'
    }}
  }});
</script>
</body>
</html>"""


# ── Page ──────────────────────────────────────────────────────────────────────
def render():
    _inject_styles()
    page_header("System architecture", "End-to-end flow from data to deployment.")

    # ── Milestone 2: animated agentic runtime (HTML/CSS) ───────────────────────
    st.markdown('<p class="arch-eyebrow">00 / Agentic runtime</p>', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="arch-heading">Milestone 2 · ML, RAG, and LLM in one pipeline</h2>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Interactive diagram: highlight the **ML-only shortcut** vs the **full agent path** "
        "(query planning → FAISS retrieval → verification → structured report)."
    )
    path_choice = st.radio(
        "Path emphasis",
        ["ML only mode", "Agent mode"],
        horizontal=True,
        key="arch_milestone2_path",
        label_visibility="collapsed",
    )
    st.markdown(render_architecture_toggle_caption())
    mode_key = "agent" if path_choice == "Agent mode" else "ml_only"
    st.markdown(f"<style>{architecture_pipeline_css()}</style>", unsafe_allow_html=True)
    st.markdown(build_pipeline_markup(mode_key), unsafe_allow_html=True)

    st.markdown('<hr class="arch-rule">', unsafe_allow_html=True)

    # ── Section 1: Mermaid diagrams ──
    st.markdown('<p class="arch-eyebrow">01 / Pipeline diagrams</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="arch-heading">High-level workflow</h2>', unsafe_allow_html=True)

    training_mermaid = """
flowchart LR
    subgraph DATA["&#x2002;Data&#x2002;"]
        A[Fake.csv] --> M[Merge]
        B[True.csv] --> M
        M --> L[Label 0/1]
    end
    subgraph PREP["&#x2002;Preprocess&#x2002;"]
        L --> C[clean_text]
        C --> D[TF-IDF]
    end
    subgraph MODELS["&#x2002;Models&#x2002;"]
        D --> E[LR]
        D --> F[NB]
        D --> G[RF]
        D --> H[SVM]
        E --> I[Best F1]
        F --> I
        G --> I
        H --> I
    end
    I --> O[pipeline.pkl]
    """

    inference_mermaid = """
flowchart TB
    U[User input] --> N[normalize]
    N --> M[ml_classify TF-IDF + model]
    M --> R{ml_confidence < threshold?}
    R -->|Yes| Q[plan_queries Groq]
    Q --> F[retrieve FAISS]
    F --> V[verify Groq or HF]
    V --> Rep[report + ui_report]
    R -->|No / error| Rep
    Rep --> O[Verdict + structured fields]
    """

    st.components.v1.html(
        _mermaid_html(training_mermaid, height=280, label="Training pipeline"),
        height=330, scrolling=False
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.components.v1.html(
        _mermaid_html(inference_mermaid, height=320, label="Runtime · LangGraph + ML branch"),
        height=380, scrolling=False
    )

    st.markdown('<hr class="arch-rule">', unsafe_allow_html=True)

    # ── Section 2: End-to-end text flow ──
    st.markdown('<p class="arch-eyebrow">02 / Data source</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="arch-heading">End-to-end flow</h2>', unsafe_allow_html=True)

    st.markdown(
        '<a class="ds-pill" href="https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset" target="_blank">'
        '<span class="icon">⬡</span> Kaggle · Fake and Real News Dataset</a>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="flow-steps">
          <span class="flow-step">Fake.csv + True.csv</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step">Merge &amp; Label</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step">clean_text()</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step">TF-IDF</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step">Train LR / NB / RF / SVM</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step highlight">pipeline.pkl</span>
        </div>
        <br>
        <div class="flow-steps">
          <span class="flow-step">User input</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step">normalize + ml_classify</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step">conditional</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step">plan → RAG → verify</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step highlight">report + verdict</span>
        </div>
        <p style="font-size:12px;color:#64748b;margin-top:10px;font-family:'DM Mono',monospace;">
          Low ML confidence triggers RAG + LLM verification; high confidence skips to report.
          Live Prediction uses the classifier path only; Deep Analysis runs the full graph (UI may force RAG).
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="arch-rule">', unsafe_allow_html=True)

    # ── Section 3: Training vs Inference cards ──
    st.markdown('<p class="arch-eyebrow">03 / Runtime separation</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="arch-heading">Training vs inference</h2>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    best_algo = get_model_algorithm_display()

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(
            f"""
            <div class="arch-card arch-card-accent">
              <div class="arch-card-title"><span class="dot"></span>Training &nbsp;·&nbsp; offline</div>
              <div class="file-list">
                <div class="file-item">
                  <span class="arrow">▸</span>
                  <div><span class="file-name">src/data/loader.py</span><br>
                  <span class="file-desc">Load Fake.csv + True.csv, extract features &amp; labels</span></div>
                </div>
                <div class="file-item">
                  <span class="arrow">▸</span>
                  <div><span class="file-name">src/features/preprocessing.py</span><br>
                  <span class="file-desc">clean_text, prepare_text_column</span></div>
                </div>
                <div class="file-item">
                  <span class="arrow">▸</span>
                  <div><span class="file-name">src/models/pipelines.py</span><br>
                  <span class="file-desc">build_lr / nb / rf / svm _pipeline</span></div>
                </div>
                <div class="file-item">
                  <span class="arrow">▸</span>
                  <div><span class="file-name">src/evaluation/</span><br>
                  <span class="file-desc">Metrics, confusion matrix, ROC, comparison table</span></div>
                </div>
                <div class="file-item">
                  <span class="arrow">▸</span>
                  <div><span class="file-name">model/pipeline.pkl</span><br>
                  <span class="file-desc">Serialised best-F1 pipeline output</span></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="arch-card">
              <div class="arch-card-title"><span class="dot" style="background:#64748b"></span>Inference &nbsp;·&nbsp; this dashboard</div>
              <div class="file-list">
                <div class="file-item">
                  <span class="arrow">▸</span>
                  <div><span class="file-name">{MODEL_DIR_NAME}/{MODEL_FILENAME}</span><br>
                  <span class="file-desc">Loaded once, cached in session</span></div>
                </div>
                <div class="file-item">
                  <span class="arrow">▸</span>
                  <div><span class="file-name">src/features/preprocessing.py</span><br>
                  <span class="file-desc">clean_text only — same contract as training</span></div>
                </div>
                <div class="file-item">
                  <span class="arrow">▸</span>
                  <div><span class="file-name">Pipeline</span><br>
                  <span class="file-desc">TF-IDF transform → {best_algo} predict / proba</span></div>
                </div>
                <div class="file-item">
                  <span class="arrow">▸</span>
                  <div><span class="file-name">src/agent/graph.py</span><br>
                  <span class="file-desc">LangGraph: normalize → ML → branch → report (optional RAG path)</span></div>
                </div>
                <div class="file-item">
                  <span class="arrow">▸</span>
                  <div><span class="file-name">src/agent/llm_service.py</span><br>
                  <span class="file-desc">Groq primary; Hugging Face inference API when configured</span></div>
                </div>
                <div class="file-item">
                  <span class="arrow">▸</span>
                  <div><span class="file-name">No training code</span><br>
                  <span class="file-desc">Zero sklearn training at dashboard runtime</span></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="arch-rule">', unsafe_allow_html=True)

    # ── Section 4: Repo table ──
    st.markdown('<p class="arch-eyebrow">04 / Codebase</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="arch-heading">Repository mapping <code style="font-size:16px;font-weight:300;">src/</code></h2>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <table class="repo-table">
          <thead>
            <tr>
              <th>Folder</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>src/data/</td>
              <td>Dataset loading (Fake.csv, True.csv), feature &amp; target extraction</td>
            </tr>
            <tr>
              <td>src/features/</td>
              <td>Text preprocessing (clean_text) — shared by train &amp; inference</td>
            </tr>
            <tr>
              <td>src/models/</td>
              <td>Pipeline definitions (LR, NB, RF, SVM) — train time only</td>
            </tr>
            <tr>
              <td>src/evaluation/</td>
              <td>Metrics, ROC, confusion matrix, results_loader, plotly_viz</td>
            </tr>
            <tr>
              <td>src/app/</td>
              <td>Streamlit dashboard, pages, core (load_model, prediction)</td>
            </tr>
            <tr>
              <td>src/agent/</td>
              <td>LangGraph workflow, nodes (normalize, ml_classify, plan_queries, retrieve, verify, report), state</td>
            </tr>
            <tr>
              <td>data/rag/</td>
              <td>FAISS index + chunk metadata for retrieval (build via scripts when needed)</td>
            </tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="arch-rule">', unsafe_allow_html=True)

    # ── Section 5: Repo structure Mermaid ──
    st.markdown('<p class="arch-eyebrow">05 / Structure</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="arch-heading">Repo structure</h2>', unsafe_allow_html=True)

    repo_mermaid = """
flowchart LR
    subgraph ROOT["Repo root"]
        DS[dataset/]
        APP[app.py]
        NB[notebook]
        SCRIPT[run_evaluation.py]
        MD["model/pipeline.pkl"]
    end
    subgraph SRC["src/"]
        DATA[data/loader]
        FEAT[features/preprocessing]
        MOD[models/pipelines]
        EVAL[evaluation/]
        APP_P[app/]
    end
    DS --> DATA
    DATA --> FEAT
    FEAT --> MOD
    MOD --> EVAL
    EVAL --> MD
    MD --> APP_P
    NB --> DATA
    SCRIPT --> DATA
    APP --> APP_P
    """

    st.components.v1.html(
        _mermaid_html(repo_mermaid, height=260, label="Full repository structure"),
        height=400, scrolling=True,
    )