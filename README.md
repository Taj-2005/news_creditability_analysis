<div align="center">

<img src="https://img.shields.io/badge/AI%2FML-Project%2011-6366F1?style=for-the-badge&logoColor=white" alt="Project Badge"/>

# News Credibility Classification System

> **Intelligent misinformation detection** combining classical NLP, retrieval-augmented generation, and LLM reasoning — from TF-IDF baselines to a full agentic fact-checking pipeline.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflows-1C3C3C?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20API-F55000?style=flat-square&logo=groq&logoColor=white)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Index-0096D6?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-MiniLM-FF6F00?style=flat-square)](https://www.sbert.net/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Inference%20API-FFD21E?style=flat-square&logo=huggingface&logoColor=000)](https://huggingface.co/)
[![Plotly](https://img.shields.io/badge/Plotly-Analytics-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE.md)
[![Dataset](https://img.shields.io/badge/Dataset-Fake%20%26%20Real%20News%20(Kaggle)-6366F1?style=flat-square)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

**[🚀 Live Demo](https://news-creditability.streamlit.app/)** &nbsp;·&nbsp; **[▶️ Video Walkthrough](https://youtu.be/U1Nnd-8Odbs?si=5IFfdhePuDTRlmCg)** &nbsp;·&nbsp; **[📑 Research paper](docs/research_paper.pdf)** &nbsp;·&nbsp; **[📄 Project brief](docs/Project_11_AI_ML.pdf)**

---

[Overview](#overview) · [Architecture](#system-architecture) · [Quickstart](#quickstart) · [Local Setup](#local-setup-ml-rag-agent) · [ML Pipeline](#ml-pipeline) · [Results](#results) · [Deployment](#deployment) · [Limitations](#limitations) · [M1 checklist](#milestone-1-deliverables-checklist) · [M2 checklist](#milestone-2-deliverables-checklist)

</div>

---

## Overview

Misinformation spreads faster than manual fact-checking can scale. This repository delivers a two-milestone system that grows from a high-performing classical ML classifier into a fully agentic, RAG-backed fact-checking pipeline — all accessible through a polished Streamlit interface.

### Milestone 1 — Classical ML Core

An end-to-end pipeline that classifies English news articles as **Fake** or **Real** using **TF-IDF** (unigrams + bigrams) with four competing models: **Logistic Regression**, **Naive Bayes**, **Random Forest**, and **SVM**. The best model by test **F1** is serialised to `model/pipeline.pkl` for fast, GPU-free inference.

### Milestone 2 — Agentic AI Layer (in-repo)

A **LangGraph** workflow in `src/agent/` extends the ML core with:

| Component | Details |
|-----------|---------|
| **Orchestration** | `normalize → ml_classify → (optional) plan_queries → retrieve → verify → report → validate_report` (bounded report retry) |
| **RAG** | MiniLM embeddings + **FAISS** (default) or optional **Chroma**; **similarity** or **MMR** retrieval (`src/rag/`, `data/rag/`) |
| **LLM Reasoning** | **Groq** primary (`GROQ_API_KEY`); optional **Gemini** (`GEMINI_API_KEY`); Gemini may **fall back to Groq** when configured |
| **UI** | Streamlit **Deep Analysis**: **Agent runtime** expander (backend / retrieval mode / LLM), live node trace, structured `final_report`, optional local feedback JSONL |

#### Structured credibility report (API + UI)

Each agent run produces a `final_report` object with: **summary** (article overview), **risk_factors**, **fact_checks** (cross-source verification rows: supported / contradicted / unknown), **verdict** (Fake / Real), **confidence** (human-readable line with probabilities), **sources** (RAG excerpts), **credibility_score** (**High** or **Low** — rubric-style trust signal from ML + evidence tension), **pattern_detection_summary** (one paragraph: classifier signal, bucket counts, retrieval usage), and **disclaimer** (ethical / limitations text also shown on the Deep Analysis page).

**Scope note:** the app performs **on-demand article analysis**, not continuous feed monitoring or alerting.

#### Hallucination mitigation & grounding

| Technique | Where |
|-----------|--------|
| **Structured JSON verification** | `verify.py`: single JSON object, fixed keys (`supported`, `contradicted`, `unknown`); **temperature 0.0** for the verifier LLM call. |
| **Evidence-first instructions** | Prompts tell the model evidence may be partial; use **unknown** when uncertain; ML label passed only as **auxiliary**, not ground truth. |
| **Deterministic cleanup** | `verify.py` parses, caps list lengths, dedupes, and fills safe fallbacks if JSON or API fails. |
| **Grounded narrative** | `ui_report.py` summary prompt: base text strictly on a **FACTS_JSON** payload; **do not invent URLs**; weak evidence must be stated clearly. |
| **RAG grounding** | Retrieved passages are the only corpus the verifier compares against (sample index under `data/rag/` — not live web crawl). |

> **Course alignment:** formal problem statement, milestones, and rubric are in [`docs/Project_11_AI_ML.pdf`](docs/Project_11_AI_ML.pdf). The [M1](#milestone-1-deliverables-checklist) / [M2](#milestone-2-deliverables-checklist) checklists below map every implementation artefact to that brief.

### Dataset at a Glance

| Attribute | Detail |
|-----------|--------|
| **Source** | Kaggle — Fake and Real News Dataset |
| **Total Records** | 40,000+ rows |
| **Class Distribution** | ~50% Fake · ~50% Real |
| **Format** | CSV (`Fake.csv`, `True.csv`) |
| **Text Fields** | Title + article body (English only) |
| **Labels** | `0` = Fake · `1` = Real (assigned automatically by loader) |

---

## Project Team — Section A

| Team Member | Role | Contribution | GitHub |
|-------------|------|-------------|--------|
| **Shaik Tajuddin** | Project Lead & GitHub Manager | Project leadership, repository design & enhancement, roadmap planning, notebook architecture, requirements and tech-stack planning, PR reviews, version control | [@Taj-2005](https://github.com/Taj-2005) |
| **Nipun** | Backend & Packaging Engineer | Folder architecture, converting notebooks into modular Python files, reusable ML pipeline, project modularisation for Streamlit integration | [@nipun1803](https://github.com/nipun1803) |
| **Hadole** | Deployment & UI Engineer | Streamlit application design, deployment architecture, model integration into UI, hosting setup and CI/CD pipeline, usability flow | [@omkar-hadole](https://github.com/omkar-hadole) |

---

## System Architecture

The system has three distinct execution paths that share the same core preprocessing and model artefacts:

| Path | Trigger | Components |
|------|---------|------------|
| **Training** | `run_evaluation.py` / notebook | Data loading → preprocessing → TF-IDF → train × 4 models → evaluate → save |
| **Fast Inference** | Live Prediction Lab | `pipeline.pkl` → `clean_text` → `predict_proba` → verdict + confidence |
| **Agent Inference** | Deep Analysis / `invoke_credibility_agent()` | LangGraph + optional RAG (FAISS/Chroma, similarity/MMR) + LLM (Groq/Gemini) + `validate_report` |

### Training Pipeline

```
┌─────────────────────────────────────────────────┐
│  Data Source                                    │
│  dataset/Fake.csv + dataset/True.csv            │
│  Fields: title, text → label (0=Fake, 1=Real)  │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  Preprocessing                                  │
│  Lowercase · strip URLs/punctuation             │
│  NLTK stopword removal · tokenise               │
│  WordNet lemmatisation                          │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  Feature Engineering (TF-IDF)                   │
│  max_features: 20K–25K · ngram: (1,2)           │
│  sublinear_tf: True                             │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  Model Training                                 │
│  Logistic Regression · Naive Bayes              │
│  Random Forest · SVM (LinearSVC)                │
│  Best by F1 → model/pipeline.pkl               │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  Evaluation                                     │
│  Precision / Recall / F1 · ROC-AUC              │
│  Confusion Matrix · 5-Fold CV F1                │
│  → model/evaluation_results.json               │
└─────────────────────────────────────────────────┘
```

### Fast Inference Pipeline

```
User Input (article text)
         │
         ▼
  Load pipeline.pkl
         │
         ▼
  clean_text()  ── same function as training; no preprocessing drift
         │
         ▼
  TF-IDF transform  →  predict_proba()
         │
         ▼
┌─────────────────────────────────┐
│  Credibility Assessment Output  │
│  Verdict:     Fake / Real       │
│  Confidence:  0.00 – 1.00       │
│  Risk Level:  High / Low        │
└─────────────────────────────────┘
```

### Deep Analysis — Agent Path

When a user triggers **Deep Analysis** (or calls `invoke_credibility_agent()`), the LangGraph workflow executes:

```
raw_text
   → normalize (clean_text)
   → ml_classify (pipeline.pkl)
   → [if confidence < threshold]
       → plan_queries (LLM: Groq / Gemini)
       → retrieve (MiniLM + FAISS or Chroma · similarity or MMR)
       → verify (LLM → structured JSON; Gemini may fall back to Groq)
   → report (build_ui_final_report)
   → validate_report (schema guard; at most one extra report attempt)
   → UI (summary · risk_factors · fact_checks · sources · disclaimer)
```

**Deep Analysis (Streamlit)** uses a high threshold so the **plan → retrieve → verify** branch **always runs** for demos; the generic graph still **skips** that branch when ML confidence is high (then **`report → validate_report`** only). Configure **RAG backend**, **MMR**, and **LLM provider** under **Deep Analysis → Agent runtime**.

### Data Flow Summary

| Stage | Input | Output | Module |
|-------|-------|--------|--------|
| Load | `Fake.csv`, `True.csv` | Merged DataFrame with labels | `src/data/loader.py` |
| Prepare | Combined title + text | `cleaned_text`, `label` columns | `src/features/preprocessing.py` |
| Split | X (text), y (label) | Stratified 80/20 train/test sets | `sklearn.model_selection` |
| Vectorise | Text series | Sparse TF-IDF matrix | `Pipeline` (fit on train only) |
| Train | TF-IDF matrix + labels | Fitted Pipeline artefact | `src/models/pipelines.py` |
| Evaluate | Pipeline + test set | Classification report, ROC-AUC, CM | `src/evaluation/` |
| Serialise | Fitted pipeline | `model/pipeline.pkl` | `joblib` |
| Serve — ML | Raw user text | Verdict + probability | `src/app/pages/live_prediction.py` |
| Serve — Agent | Raw user text | `final_report` + graph state | `src/app/pages/deep_analysis.py` |

> **Critical invariant:** `clean_text()` and the fitted `pipeline.pkl` are shared between training and all inference paths to eliminate preprocessing drift.

---

## Repository Structure

```
news_creditability_analysis/
├── README.md
├── LICENSE.md                          # MIT licence + contributor list
├── requirements.txt
├── .env.example                        # Template for GROQ / optional GEMINI / LLM_PROVIDER (never commit .env)
├── app.py                              # Streamlit entry point
│
├── .streamlit/
│   ├── config.toml                     # Theme configuration
│   └── secrets.example.toml           # Streamlit Cloud secrets template
│
├── docs/
│   ├── Project_11_AI_ML.pdf           # Course brief, rubric, and submission notes
│   └── research_paper.pdf             # Project research write-up
│
├── notebook/
│   └── news_credibility.ipynb         # Full pipeline: EDA → preprocess → train → evaluate → save
│
├── scripts/
│   ├── run_evaluation.py              # Train all models, select best by F1, save artefacts
│   ├── build_rag_index.py             # Build MiniLM + FAISS index under data/rag/
│   └── build_chroma_store.py          # Optional: persist Chroma store under data/rag/chroma_store/
│
├── dataset/
│   ├── Fake.csv                       # Kaggle fake news corpus
│   └── True.csv                       # Kaggle real news corpus
│
├── model/
│   ├── pipeline.pkl                   # Best model (TF-IDF + classifier)
│   └── evaluation_results.json        # Metrics and dataset stats (generated at runtime)
│
├── plots/                             # EDA and evaluation figures
│
├── data/
│   └── rag/                           # faiss.index + chunks.json (+ optional chroma_store/ from build_chroma_store.py)
│
└── src/
    ├── config/
    │   └── env_bootstrap.py           # Dotenv + Streamlit secrets merge on startup
    ├── data/
    │   └── loader.py                  # load_dataset() — merges CSVs and assigns labels
    ├── features/
    │   └── preprocessing.py           # clean_text(), prepare_text_column() — shared train/inference
    ├── models/
    │   └── pipelines.py               # build_lr/nb/rf/svm_pipeline()
    ├── evaluation/
    │   ├── metrics.py
    │   ├── results_loader.py          # Loads evaluation_results.json for dashboard
    │   ├── plotly_viz.py              # ROC, PR curve, confusion matrix, gauge charts
    │   └── visualization.py
    │
    ├── agent/                         # Milestone 2 — LangGraph + LLM + RAG
    │   ├── state.py                   # AgentState + DEFAULT_LOW_CONFIDENCE_THRESHOLD
    │   ├── graph.py                   # build_graph(), invoke_credibility_agent(); validate_report + retry edge
    │   ├── llm_service.py             # Groq / optional Gemini; Gemini → Groq fallback when configured
    │   ├── feedback.py                # Optional JSONL feedback logger (Deep Analysis)
    │   ├── ui_report.py               # build_ui_final_report() → UI dict
    │   └── nodes/
    │       ├── normalize.py           # clean_text wrapper
    │       ├── ml_classify.py         # TF-IDF + classifier via core
    │       ├── plan_queries.py        # LLM RAG query planning (fallback heuristics)
    │       ├── retrieve.py            # FAISS or Chroma; similarity or MMR
    │       ├── verify.py              # LLM → JSON {supported, contradicted, unknown}
    │       ├── report.py              # Assemble final_report via ui_report
    │       └── validate_report.py     # Schema validation; triggers bounded report retry
    │
    ├── rag/                           # Local RAG — MiniLM + FAISS (optional Chroma)
    │   ├── embeddings.py              # sentence-transformers MiniLM, L2-normalised vectors
    │   ├── store.py                   # FAISS IndexFlatIP + chunks.json persistence
    │   ├── chroma_store.py            # Optional persistent Chroma collection
    │   └── retrieve.py                # Similarity or MMR; backend switch (faiss / chroma)
    │
    ├── utils/
    └── app/
        ├── core.py                    # load_model(), run_prediction(), validation
        ├── dashboard.py               # Multi-page Streamlit app with sidebar navigation
        ├── main.py                    # Alternate entry point
        ├── components/
        │   ├── styles.py              # Application CSS and theming
        │   ├── ui.py                  # Shared UI components (page_header, etc.)
        │   ├── agent_pipeline.py      # Deep Analysis + Live Prediction stepped pipeline UI
        │   └── architecture_flow.py   # Animated ML vs. agent architecture diagram (HTML/CSS)
        └── pages/
            ├── home.py                # Overview — KPIs, dataset summary, dual-milestone pitch
            ├── dataset_insights.py    # Class distribution, text lengths, TF-IDF features
            ├── model_compare.py       # ROC, PR curve, confusion matrix, feature importance
            ├── live_prediction.py     # Fast ML-only verdict + confidence gauge
            ├── deep_analysis.py       # LangGraph stream, Agent runtime controls, feedback, structured report
            └── architecture.py        # Mermaid diagrams, animated pipeline, repo map
```

---

## Quickstart

> **Note:** The repo may include a pre-trained `model/pipeline.pkl`. If so, you can skip steps 3–4 and go straight from installation to launching the app. For the full **RAG + LangGraph + LLM** stack, follow [Local Setup](#local-setup-ml-rag-agent) after installing dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/Taj-2005/news_creditability_analysis
cd news_creditability_analysis
```

### 2. Set Up the Environment

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the Dataset

Download the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle and place both files under `dataset/`:

```
dataset/
├── Fake.csv
└── True.csv
```

Columns used: `title`, `text`. Labels are assigned automatically (`Fake = 0`, `Real = 1`).

### 4. Train the Model

**Option A — Jupyter Notebook** *(full EDA, plots, and model comparison)*

```bash
jupyter notebook notebook/news_credibility.ipynb
# Run from the repo root so dataset/ and src/ are on the Python path.
# Outputs: model/pipeline.pkl  model/evaluation_results.json  plots/*.png
```

**Option B — Script** *(fast training and artefact generation)*

```bash
python scripts/run_evaluation.py
# Trains LR, Naive Bayes, Random Forest, and SVM.
# Saves the best model by F1 as model/pipeline.pkl and model/evaluation_results.json.
```

### 5. Launch the Application

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

The **News Credibility Analyzer** dashboard includes six pages:

| Page | Description |
|------|-------------|
| **Overview** | KPIs, dataset summary, dual-milestone system pitch |
| **Dataset Intelligence** | Class distribution, text lengths, top TF-IDF features |
| **Model Comparison** | ROC curves, PR curves, confusion matrices, CV F1 chart |
| **Live Prediction Lab** | Fast ML-only inference — verdict + confidence gauge |
| **Deep Analysis** | Full LangGraph agent — Agent runtime (FAISS/Chroma, similarity/MMR, Groq/Gemini), live trace, structured report |
| **Architecture** | Training/runtime Mermaid diagrams, animated pipeline, repo map |

---

## Local Setup (ML, RAG, Agent)

Use this checklist after completing Quickstart steps 1–2. **Order matters** where noted.

### Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10–3.12 recommended (3.14 may exhibit LangChain/Pydantic warnings) |
| **Disk** | ~500 MB – 1 GB for `sentence-transformers` + first-time MiniLM download |
| **Network** | Required once to download `all-MiniLM-L6-v2` from Hugging Face; Groq calls require outbound HTTPS |

### A. ML Model (`model/pipeline.pkl`)

If the repo already includes `model/pipeline.pkl`, skip this step. To regenerate:

1. Place `Fake.csv` / `True.csv` under `dataset/` (see [Quickstart §3](#3-add-the-dataset)).
2. From the repository root, with the virtual environment activated:

```bash
python scripts/run_evaluation.py
# Outputs: model/pipeline.pkl  model/evaluation_results.json
```

Alternatively, run `notebook/news_credibility.ipynb` from the repo root for the full analysis and plots.

### B. RAG Index (MiniLM + FAISS)

```bash
python scripts/build_rag_index.py
# Downloads all-MiniLM-L6-v2 on first run (cached to .cache/huggingface/)
# Outputs: data/rag/faiss.index  data/rag/chunks.json
# Prints a short retrieval smoke-test to confirm the index is working.
```

If `data/rag/` is missing when the agent runs, the `retrieve` node completes but records a `rag_error` in the report. Re-run this script whenever you change chunking logic or swap in a custom corpus (edit `SAMPLE_DOCUMENTS` in the script).

### B2. Optional ChromaDB Store (alternative backend)

If you want a Chroma-backed vector store (optional), build it after FAISS:

```bash
python scripts/build_chroma_store.py
# Outputs: data/rag/chroma_store/
```

On the **Deep Analysis** page, open **Agent runtime** and set **RAG backend** to `chroma`. If `data/rag/chroma_store/` is missing, the agent continues but records a `rag_error`.

### C. Groq API Key (LLM Reasoning)

1. Generate a key at [Groq Console](https://console.groq.com/keys).
2. Copy `.env.example` to `.env` and add your key — `python-dotenv` loads it automatically.

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=<your-key>
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | **Yes** (for LLM output) | Secret key from Groq Console |
| `GROQ_MODEL` | No | Chat model ID; defaults to `llama-3.1-8b-instant` |

**Graceful degradation without a key:** `plan_queries` falls back to text-window queries; `verify` returns deterministic `unknown` entries (with `llm_error` set); `report` still populates all structured fields, though `llm_summary` may be `null`.

#### Optional Gemini Backend (with fallback to Groq)

```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=...
```

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | No | `auto` (default), `groq`, or `gemini` |
| `GEMINI_API_KEY` | Yes (Gemini) | Gemini API key |
| `GEMINI_MODEL` | No | Defaults to `gemini-1.5-flash` |

If Gemini is selected and fails at runtime, the agent **falls back to Groq** when `GROQ_API_KEY` is set.

### D. LangGraph Agent

With `model/pipeline.pkl`, `data/rag/`, and `GROQ_API_KEY` in place, test the full agent from the command line:

```bash
python -c "
from src.agent.graph import invoke_credibility_agent
import json
out = invoke_credibility_agent(
    'WASHINGTON (Reuters) — The Federal Reserve left interest rates unchanged.',
    confidence_threshold=0.65,
)
print(json.dumps(out.get('final_report', {}), indent=2, ensure_ascii=False)[:1200])
"
```

- **`confidence_threshold`**: articles whose predicted-class probability falls **below** this value trigger the full `plan_queries → retrieve → verify → report` path; others go straight to `report`. Default is `DEFAULT_LOW_CONFIDENCE_THRESHOLD` in `src/agent/state.py`.
- You may also call `build_graph().invoke({"raw_text": "..."})` directly.

### E. One-Shot Setup (Copy-Paste)

```bash
# 0. Clone and install (see Quickstart §1–2)

# 1. Configure environment
cp .env.example .env          # then set GROQ_API_KEY in .env

# 2. Train the ML model (requires dataset/ CSVs)
python scripts/run_evaluation.py

# 3. Build the RAG index (downloads MiniLM on first run)
python scripts/build_rag_index.py

# 4. Launch the app
streamlit run app.py          # http://localhost:8501
```

---

## ML Pipeline

### Stage 1 — Text Preprocessing

Every document passes through the same deterministic `clean_text()` function at both training time and inference time, ensuring zero preprocessing drift.

```
Raw string
  → lowercase
  → remove URLs (http / https / www)
  → remove @mentions
  → keep only [a-z] and whitespace
  → tokenise (split on whitespace)
  → remove NLTK English stopwords
  → remove tokens with length ≤ 2
  → WordNet lemmatisation
  → rejoin tokens with a single space
```

```python
# src/features/preprocessing.py
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)
```

### Stage 2 — Feature Extraction (TF-IDF)

| Hyperparameter | Value |
|----------------|-------|
| `max_features` | 20,000 – 25,000 (model-dependent) |
| `ngram_range` | `(1, 2)` — unigrams and bigrams |
| `min_df` | 2 |
| `max_df` | 0.92 |
| `sublinear_tf` | `True` |

### Stage 3 — Classification Models

| Model | Key Hyperparameters | Notes |
|-------|---------------------|-------|
| **Logistic Regression** | `C=2.0`, `class_weight='balanced'`, `solver='lbfgs'` | Interpretable; calibrated probabilities |
| **Naive Bayes** | `MultinomialNB(alpha=0.1)` | Fast; performs well on sparse TF-IDF matrices |
| **Random Forest** | `n_estimators=200`, `max_depth=30`, `class_weight='balanced'` | Non-linear; robust to feature noise |
| **SVM** | `LinearSVC(C=1.0)`, `class_weight='balanced'` | Strong linear decision boundary |

The best model by F1 score is saved as `model/pipeline.pkl`. All four models are compared in the notebook and in the **Model Comparison** dashboard page.

### Stage 4 — Evaluation

```python
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score

y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1', n_jobs=-1)
print(f"5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### Usage from Python

```python
import joblib
from src.features.preprocessing import clean_text

pipeline = joblib.load("model/pipeline.pkl")

text = "Your news headline or article body here..."
cleaned = clean_text(text)

label = pipeline.predict([cleaned])[0]          # 0 = Fake, 1 = Real
proba = pipeline.predict_proba([cleaned])[0]    # [P(Fake), P(Real)]

print(f"Verdict:          {'Fake' if label == 0 else 'Real'}")
print(f"Fake probability: {proba[0]:.2%}")
```

---

## Results

> Results are generated by the notebook or `python scripts/run_evaluation.py` and persisted to `model/evaluation_results.json`. The Streamlit dashboard reads from this file — no metrics are hardcoded.

| Metric | Logistic Regression | Naive Bayes | Random Forest | SVM |
|--------|---------------------|-------------|---------------|-----|
| **Accuracy** | High | Competitive | Competitive | High |
| **Precision / Recall / F1** | Strong | Strong | Strong | Strong |
| **ROC-AUC** | High | Good | Good | High |
| **5-Fold CV F1** | *See artefact* | *See artefact* | *See artefact* | *See artefact* |

The best model by F1 is automatically selected and saved. The notebook and **Model Comparison** page display full ROC curves, Precision-Recall curves, confusion matrices, and a CV F1 bar chart for all four models.

### Metric Definitions

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Precision (Fake)** | Of all articles flagged Fake, the fraction that are actually Fake | Reduces false alarms on real news |
| **Recall (Fake)** | Of all actual Fake articles, the fraction correctly caught | Reduces missed misinformation |
| **F1 Score** | Harmonic mean of Precision and Recall | Primary optimisation target |
| **ROC-AUC** | Ranking quality across all decision thresholds | Threshold-independent performance measure |

For misinformation detection, both Precision and Recall are critical: high Recall catches more fake news, while high Precision avoids incorrectly flagging legitimate articles.

---

## Application

### Streamlit Dashboard — News Credibility Analyzer

**[🌐 Live Demo](https://news-creditability.streamlit.app)** &nbsp;·&nbsp; Run locally: `streamlit run app.py` → `http://localhost:8501`

| Page | Description |
|------|-------------|
| **Overview** | Key metrics (accuracy, F1, ROC-AUC, 5-fold CV F1), dataset summary, dual-milestone attribution |
| **Dataset Intelligence** | Class distribution, text length distributions, top TF-IDF features per class |
| **Model Comparison** | ROC curves, Precision-Recall curves, confusion matrices, CV F1 distribution, linear-model feature importance |
| **Live Prediction Lab** | ML-only: paste text → Fake/Real verdict + confidence score + gauge |
| **Deep Analysis** | Full agent: summary, risk factors, RAG sources, structured fact-checks; **Agent runtime** expander for **FAISS/Chroma**, **similarity/MMR**, and **Groq/Gemini** (Gemini falls back to Groq when configured) + optional feedback logging to `data/feedback/feedback.jsonl` |
| **Architecture** | Training/runtime Mermaid diagrams, animated Milestone 2 pipeline, full repo structure map |

The app reads `model/pipeline.pkl` and `model/evaluation_results.json` at runtime. If either file is absent, the relevant page displays a clear prompt to run the training script — no stale or hardcoded data is ever shown.

---

## Deployment

This project is structured for **Streamlit Community Cloud** (free tier). Local-only deployments do not satisfy project requirements.

### Deploy to Streamlit Community Cloud

**Step 1 — Push to GitHub**

```bash
git add .
git commit -m "feat: milestone complete"
git push origin main
```

> If `model/pipeline.pkl` exceeds 100 MB, use Git LFS:
> ```bash
> git lfs install && git lfs track "*.pkl"
> git add .gitattributes && git commit -m "chore: add git lfs"
> ```

**Step 2 — Connect to Streamlit**

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New App** → select your repository.
3. Set **Branch:** `main` and **Main file:** `app.py`.
4. Under **Advanced settings → Secrets**, add your keys as TOML — at minimum `GROQ_API_KEY` (see `.streamlit/secrets.example.toml` for flat or nested `[groq]`). Add `GEMINI_API_KEY` / `LLM_PROVIDER` only if you use Gemini.
5. Click **Deploy**.

> **RAG index for Deep Analysis:** The deployed app expects `data/rag/faiss.index` and `data/rag/chunks.json` to be committed to the repository. Streamlit Cloud does not run `build_rag_index.py` for you — build the index locally and commit the output files before deploying.
>
> **Optional Chroma backend:** If you want **RAG backend = chroma** in production (Deep Analysis → Agent runtime), also commit `data/rag/chroma_store/` (build locally via `python scripts/build_chroma_store.py`).

**Step 3 — Verify**

Your live application URL will appear in the Streamlit Cloud dashboard after the build completes.

### Environment Requirements

| Requirement | Value |
|-------------|-------|
| **Python** | 3.8 or higher |
| **GPU** | Not required |
| **RAM** | ~512 MB baseline (ML only); +300–800 MB when loading MiniLM for RAG or first agent run |
| **Storage** | ~100–200 MB for `pipeline.pkl`; extra for `.cache/huggingface/` and `data/rag/` |

### Dependencies

Core stack: **Streamlit**, **scikit-learn**, **NLTK**, **Plotly**. Milestone 2 additions: **sentence-transformers**, **faiss-cpu**, **langgraph**, **groq**, **python-dotenv**. `requirements.txt` uses lower bounds compatible with Streamlit Community Cloud.

```bash
pip install -r requirements.txt
```

---

## Limitations

| Area | Limitation |
|------|-----------|
| **Language** | Preprocessing uses English NLTK resources; the dataset is English-only |
| **Domain** | Trained on the Kaggle Fake and Real News corpus; may not generalise to other domains or topics |
| **Task Scope** | Binary classification only (Fake/Real); no multi-label support for satire, misleading framing, or out-of-context articles |
| **Temporal Generalisation** | Random train/test split; performance may be overstated if the news distribution shifts over time |
| **Model Ceiling** | Classical TF-IDF + LR/SVM; not state-of-the-art compared to fine-tuned transformer models |
| **Probability Calibration** | `predict_proba` outputs are not formally calibrated (no Platt scaling or isotonic regression) |
| **Agent / LLM** | LLM outputs (Groq/Gemini) depend on model and prompt; the RAG index is project-sized and is not a substitute for professional fact-checking |

---

## Future Work

**Already in this repo:** LangGraph agent (including `validate_report` + bounded retry), FAISS + optional Chroma + MMR retrieval, Groq + optional Gemini (with Groq fallback), structured verification JSON, `final_report` builder, Deep Analysis UI with Agent runtime controls, and optional local feedback logging.

**Planned / stretch goals:**

- **REST API** for batch inference
- **Multi-label taxonomies** (satire, misleading, out-of-context)
- **Probability calibration** (e.g. Platt scaling / isotonic regression)
- **Larger or live fact-check corpora** beyond the bundled sample index
- **Transformer baseline** (e.g. fine-tuned mBERT) compared to TF-IDF
- **Hardened production deployment** (rate limiting, authentication, monitoring)

---

## Milestone 1 Deliverables Checklist

> Aligned with **Project 11 — AI/ML Systems** coursework expectations. See [`docs/Project_11_AI_ML.pdf`](docs/Project_11_AI_ML.pdf) for the formal brief, rubric, and submission notes.

- [x] Problem understanding and media use-case documented
- [x] Input–output specification (`text → credibility label + probability`)
- [x] System architecture diagram (training + inference; extended for Milestone 2)
- [x] Working Streamlit application (**News Credibility Analyzer**) with multi-page UI
- [x] Model performance evaluation report (Precision · Recall · F1 · ROC-AUC · CV F1)
- [x] Multiple models trained and compared (Logistic Regression · Naive Bayes · Random Forest · SVM)
- [x] Confusion matrices, ROC curves, and Precision-Recall curves
- [x] TF-IDF feature interpretability (top fake/real indicative terms)
- [x] Dataset attribution (Kaggle Fake and Real News) in app and README
- [x] Publicly hosted application — [Streamlit Cloud demo](https://news-creditability.streamlit.app/)

## Milestone 2 Deliverables Checklist

> Extends Project 11 with an **agentic AI** layer: LangGraph orchestration, RAG (FAISS default, optional Chroma; similarity or MMR), and LLM reasoning (Groq primary, optional Gemini with Groq fallback).

- [x] **Agentic workflow** — `src/agent/graph.py`: `normalize → ml_classify → conditional → … → report → validate_report` (with `plan_queries → retrieve → verify` when confidence is below threshold; bounded report retry on validation failure)
- [x] **RAG** — `src/rag/` + `data/rag/` (`faiss.index`, `chunks.json`; optional `chroma_store/`); build via `scripts/build_rag_index.py` and optionally `scripts/build_chroma_store.py`
- [x] **LLM integration** — `src/agent/llm_service.py` (Groq; optional Gemini; HF Inference API when configured for other call sites); used for query planning, JSON verification, and narrative reporting
- [x] **Structured verification** — `supported` / `contradicted` / `unknown` buckets parsed in `src/agent/nodes/verify.py`
- [x] **UI: Deep Analysis** — full LangGraph path, **Agent runtime** expander (RAG backend, MMR, LLM provider), live stream/trace, `final_report` (summary, risks, sources, fact-checks, rubric fields, disclaimer), optional `data/feedback/feedback.jsonl`
- [x] **UI: Live Prediction Lab** — fast ML-only path (same `clean_text` + `pipeline.pkl` as the agent's ML node; no FAISS step)
- [x] **UI: Architecture** — Milestone 2 animated pipeline, training + runtime LangGraph Mermaid diagrams, repo structure map
- [x] **Configuration & deployment** — `requirements.txt`, `.env` / Streamlit secrets for `GROQ_API_KEY`, full local setup documentation
- [x] **Documentation** — README overview, architecture, and checklists cross-linked to [`docs/Project_11_AI_ML.pdf`](docs/Project_11_AI_ML.pdf)

---

<div align="center">

**Project 11 · Intelligent News Credibility Analysis and Agentic Misinformation Monitoring**

*Classical ML (scikit-learn) · RAG (FAISS/Chroma + MiniLM + MMR) · LLM (Groq/Gemini) · Streamlit*

Made with ❤️ by [Shaik Tajuddin](https://github.com/Taj-2005), [Nipun](https://github.com/nipun1803), and [Hadole](https://github.com/omkar-hadole) — Section A

</div>