<div align="center">

# News Credibility Classification System

### Intelligent Misinformation Detection via Classical NLP and Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Fake%20%26%20Real%20News%20(Kaggle)-6366F1?style=flat-square)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

**Project 11 · Milestone 1 · AI/ML Systems**

[Overview](#overview) · [Architecture](#system-architecture) · [Quickstart](#quickstart) · [Local setup (ML, RAG, agent)](#local-setup-ml-rag-agent) · [ML Pipeline](#ml-pipeline) · [Results](#results) · [Deployment](#deployment) · [Limitations](#limitations)

**Live Application:** [Demo](https://news-creditability.streamlit.app/) · **Video Walkthrough:** [Watch on YouTube](https://youtu.be/U1Nnd-8Odbs?si=5IFfdhePuDTRlmCg)

</div>

---

## Overview

Misinformation spreads faster than manual fact-checking can scale. This project builds an end-to-end, production-structured ML pipeline that automatically classifies news articles as **Fake** or **Real** using classical NLP techniques — no LLMs, no GPU required.

Trained on the **[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)** (40,000+ articles), the system uses TF-IDF vectorization (unigrams and bigrams) with Logistic Regression, Naive Bayes, Random Forest, and SVM. The best-performing model by F1 score is persisted and served through a multi-page **Streamlit** web application called the **News Credibility Analyzer**.

**This is Milestone 1** of a two-phase project. Milestone 2 will extend the system into a LangGraph-based agentic AI assistant with RAG-powered fact-checking.

### Dataset at a Glance


| Attribute          | Detail                                  |
| ------------------ | --------------------------------------- |
| Source             | Kaggle — Fake and Real News Dataset    |
| Total Records      | 40,000+ rows                            |
| Class Distribution | ~50% Fake · ~50% Real                  |
| Data Format        | CSV (Fake.csv, True.csv)                |
| Text Fields        | Title + article body (English)          |
| Labels             | Fake = 0, Real = 1 (assigned by loader) |

---

## Project Team — Section A


| Team Member        | Role                            | Contribution                                                                                                                                                                                                            | GitHub                                           |
| ------------------ | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **Shaik Tajuddin** | Project Lead and GitHub Manager | Project leadership, repository creation and enhancement, roadmap planning, repo structure design, notebook architecture, requirements and tech stack planning, PR reviews, version control and collaboration management | [@Taj-2005](https://github.com/Taj-2005)         |
| **Nipun**          | Backend and Packaging Engineer  | Folder architecture implementation, converting notebooks into modular Python files, building reusable ML pipeline, preparing codebase for Streamlit integration, project modularization                                 | [@nipun1803](https://github.com/nipun1803)       |
| **Hadole**         | Deployment and UI Engineer      | Streamlit application design, deployment architecture, model integration into UI, preparing deployable structure, hosting setup and deployment pipeline, usability flow                                                 | [@omkar-hadole](https://github.com/omkar-hadole) |

---

## System Architecture

The system is divided into two phases: a **Training Pipeline** that builds and persists the best model, and an **Inference Pipeline** that serves real-time predictions through the web application.

### Training Pipeline

```
Data Source
───────────────────────────────────────────────
Kaggle Fake and Real News Dataset
dataset/Fake.csv + dataset/True.csv
(fields: title, text, label)

          │
          ▼

┌─────────────────────────────┐
│        Data Loading         │
│  Fake.csv + True.csv        │
│  Merge → label (0=Fake,     │
│  1=Real) → shuffle          │
└─────────────┬───────────────┘
              │
              ▼

┌─────────────────────────────┐
│       Text Preparation      │
│  Concatenate title + text   │
│  Assign binary labels       │
└─────────────┬───────────────┘
              │
              ▼

┌─────────────────────────────────────────┐
│            Preprocessing                │
│  Lowercasing                            │
│  Remove URLs and punctuation            │
│  Stopword removal (NLTK English)        │
│  Tokenization                           │
│  Lemmatization (WordNet)                │
└─────────────┬───────────────────────────┘
              │
              ▼

┌─────────────────────────────────────────┐
│       Feature Engineering (TF-IDF)      │
│  max_features: 20K–25K                  │
│  ngram_range: (1, 2) — unigrams +       │
│  bigrams                                │
│  sublinear_tf: True                     │
└─────────────┬───────────────────────────┘
              │
              ▼

┌─────────────────────────────────────────┐
│        Machine Learning Models          │
│  Logistic Regression                    │
│  Naive Bayes                            │
│  Random Forest                          │
│  SVM (LinearSVC)                        │
│  Best model by F1 → pipeline.pkl        │
└─────────────┬───────────────────────────┘
              │
              ▼

┌─────────────────────────────────────────┐
│           Model Evaluation              │
│  Precision / Recall / F1                │
│  ROC-AUC / Confusion Matrix             │
│  5-Fold Cross-Validation F1             │
└─────────────┬───────────────────────────┘
              │
              ▼

┌─────────────────────────────────────────┐
│           Model Persistence             │
│  model/pipeline.pkl                     │
│  model/evaluation_results.json          │
└─────────────────────────────────────────┘
```

### Inference Pipeline

```
User Input (Article Text)
          │
          ▼

┌─────────────────────────┐
│    Load Saved Model     │
│    pipeline.pkl         │
└───────────┬─────────────┘
            │
            ▼

┌─────────────────────────┐
│    Clean Input Text     │
│  Same preprocessing     │
│  function (no drift)    │
└───────────┬─────────────┘
            │
            ▼

┌─────────────────────────┐
│     TF-IDF Transform    │
│  Vectorize cleaned text │
└───────────┬─────────────┘
            │
            ▼

┌─────────────────────────┐
│    Prediction Engine    │
│  predict_proba()        │
└───────────┬─────────────┘
            │
            ▼

┌───────────────────────────────────┐
│    Credibility Assessment Output  │
│  Verdict: Fake / Real             │
│  Confidence Score (0.00 – 1.00)   │
│  Risk Level: High / Low           │
└───────────────────────────────────┘
```

### Data Flow Summary


| Stage     | Input                                  | Output                             | Module                           |
| --------- | -------------------------------------- | ---------------------------------- | -------------------------------- |
| Load      | `dataset/Fake.csv`, `dataset/True.csv` | Merged DataFrame with labels       | `src/data/loader.py`             |
| Prepare   | Combined title + text                  | `cleaned_text`, `label` columns    | `src/features/preprocessing.py`  |
| Split     | X (text), y (label)                    | Stratified 80/20 train/test sets   | `sklearn.model_selection`        |
| Vectorize | Text series                            | Sparse TF-IDF matrix               | `Pipeline` (fit on train only)   |
| Train     | TF-IDF matrix + labels                 | Fitted Pipeline artifact           | `src/models/pipelines.py`        |
| Evaluate  | Pipeline + test set                    | Classification report, ROC-AUC, CM | `src/evaluation/`                |
| Serialize | Fitted pipeline                        | `model/pipeline.pkl`               | `joblib`                         |
| Serve     | Raw user text                          | Verdict + probability              | `src/app/` (Streamlit dashboard) |

> **Critical invariant:** The same `clean_text()` function and the same fitted pipeline (vectorizer + classifier) are used in both training and inference to eliminate preprocessing drift.

---

## Repository Structure

```
news_creditability_analysis/
├── README.md
├── requirements.txt
├── app.py                              # Streamlit entry point
├── .streamlit/
│   └── config.toml                     # Theme configuration
├── notebook/
│   └── news_credibility.ipynb          # Full pipeline: load, EDA, preprocess, train, evaluate, save
├── scripts/
│   ├── run_evaluation.py               # Train all models, select best by F1, save artifacts
│   └── build_rag_index.py              # Build local MiniLM + FAISS index under data/rag/
├── dataset/
│   ├── Fake.csv                        # Kaggle fake news corpus
│   └── True.csv                        # Kaggle real news corpus
├── model/
│   ├── pipeline.pkl                    # Best model (TF-IDF + classifier)
│   └── evaluation_results.json         # Metrics and dataset stats (generated at runtime)
├── plots/                              # EDA and evaluation figures
├── data/
│   └── rag/                            # Generated: faiss.index + chunks.json (see scripts/build_rag_index.py)
└── src/
    ├── __init__.py
    ├── data/
    │   └── loader.py                   # load_dataset() — merges CSVs and assigns labels
    ├── features/
    │   └── preprocessing.py            # clean_text(), prepare_text_column() — shared train/inference
    ├── models/
    │   └── pipelines.py                # build_lr_pipeline(), build_nb_pipeline(), build_rf_pipeline(), build_svm_pipeline()
    ├── evaluation/
    │   ├── metrics.py
    │   ├── results_loader.py            # Loads evaluation_results.json for dashboard
    │   ├── plotly_viz.py               # Plotly charts (ROC, PR curve, confusion matrix, gauge)
    │   └── visualization.py
    ├── agent/                          # Milestone 2 — LangGraph agent (does not replace ML training code)
    │   ├── state.py                    # AgentState + DEFAULT_LOW_CONFIDENCE_THRESHOLD
    │   ├── graph.py                    # build_graph(), invoke_credibility_agent() — normalize → ML → RAG? → report
    │   └── nodes/
    │       ├── normalize.py            # clean_text; ml_classify.py → core.run_prediction
    │       ├── ml_classify.py          # Calls existing TF-IDF + classifier path
    │       ├── retrieve.py             # FAISS top-k (data/rag)
    │       ├── verify.py               # Placeholder verifier (no LLM yet)
    │       └── report.py               # final_report JSON for UI
    ├── rag/                            # Local RAG — MiniLM + FAISS (Milestone 2)
    │   ├── embeddings.py               # sentence-transformers MiniLM, L2-normalized vectors
    │   ├── store.py                    # FAISS IndexFlatIP + chunks.json persistence
    │   └── retrieve.py                 # Top-k similarity search
    ├── utils/
    └── app/
        ├── core.py                     # load_model(), run_prediction(), validation
        ├── dashboard.py                # Multi-page Streamlit app with sidebar navigation
        ├── main.py                     # Alternate entry point
        ├── components/
        │   ├── styles.py               # Application CSS and theming
        │   └── ui.py                   # Shared UI components (page_header, etc.)
        └── pages/
            ├── home.py                 # Overview — KPIs and dataset summary
            ├── dataset_insights.py     # Class distribution, text length, TF-IDF features
            ├── model_compare.py        # ROC, PR curve, confusion matrix, feature importance
            ├── live_prediction.py      # Text input → Fake/Real verdict with confidence
            └── architecture.py        # Pipeline and repo mapping
```

---

## Quickstart

> This repo may include a pre-trained `model/pipeline.pkl`. You can skip steps 3–4 and open the app after step 2. For **RAG** (FAISS + MiniLM) and the **LangGraph agent**, follow [Local setup (ML, RAG, agent)](#local-setup-ml-rag-agent) after installing dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/Taj-2005/news_creditability_analysis
cd news_creditability_analysis
```

### 2. Set Up the Environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the Dataset

Download the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle and place `Fake.csv` and `True.csv` inside the `dataset/` folder.

```
dataset/
├── Fake.csv
└── True.csv
```

Columns used: `title`, `text`. Labels are assigned automatically — Fake = 0, Real = 1.

### 4. Train the Model

**Option A — Notebook (full analysis with plots and model comparison)**

```bash
jupyter notebook notebook/news_credibility.ipynb
# Run from repo root so dataset/ and src/ are on the Python path.
# Outputs: model/pipeline.pkl, model/evaluation_results.json, plots/*.png
```

**Option B — Script (fast training and artifact generation)**

```bash
python scripts/run_evaluation.py
# Trains LR, Naive Bayes, Random Forest, and SVM.
# Picks the best model by F1 and saves model/pipeline.pkl and model/evaluation_results.json.
```

### 5. Launch the Application

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

The **News Credibility Analyzer** dashboard has five pages: **Overview**, **Dataset Intelligence**, **Model Comparison**, **Live Prediction Lab**, and **Architecture**. Use the **Live Prediction Lab** to paste any news article and receive a Fake/Real verdict with a confidence score.

---

## Local setup (ML, RAG, agent)

Use this checklist after [Quickstart](#quickstart) steps 1–2 (`venv` + `pip install -r requirements.txt`). Order matters where noted.

### Prerequisites

| Item | Notes |
|------|--------|
| **Python** | 3.10–3.12 recommended (3.14 may show LangChain/Pydantic warnings). |
| **Disk** | ~500 MB–1 GB for `sentence-transformers` + first-time MiniLM download; `faiss-cpu` is small. |
| **Network** | Required once to download **all-MiniLM-L6-v2** from Hugging Face when building the RAG index. |

### A. Credibility ML model (`model/pipeline.pkl`)

The repo may already include `model/pipeline.pkl`. To regenerate metrics and the best TF-IDF classifier:

1. Place Kaggle **`Fake.csv`** / **`True.csv`** under `dataset/` (see [Quickstart §3](#3-add-the-dataset)).
2. From the **repository root**, with the venv activated:

```bash
python scripts/run_evaluation.py
```

Outputs: `model/pipeline.pkl`, `model/evaluation_results.json`.

*(Alternatively run `notebook/news_credibility.ipynb` from the repo root for the full report and plots.)*

### B. RAG index (MiniLM + FAISS under `data/rag/`)

The RAG stack lives in `src/rag/` and uses **sentence-transformers** (`all-MiniLM-L6-v2`) plus **FAISS**. The first run downloads the embedding model into **`<repo>/.cache/huggingface/`** (see `src/rag/embeddings.py`); that folder is gitignored.

From the **repository root**:

```bash
python scripts/build_rag_index.py
```

This embeds a small built-in sample corpus, writes **`data/rag/faiss.index`** and **`data/rag/chunks.json`**, and prints a short retrieval smoke test. If those files are missing, the agent’s retrieve step still completes but records a `rag_error` in the report until you run the script.

**Re-run** this script whenever you change chunking logic or swap in your own corpus (edit `SAMPLE_DOCUMENTS` in `scripts/build_rag_index.py` or extend the script to load JSON/CSV).

### C. LangGraph agent (normalize → ML → optional RAG → report)

The compiled graph is built in `src/agent/graph.py`. From the **repository root**, with `model/pipeline.pkl` (and ideally `data/rag/` from step B):

```bash
python -c "
from src.agent.graph import invoke_credibility_agent
out = invoke_credibility_agent(
    'WASHINGTON (Reuters) - The Federal Reserve left interest rates unchanged.',
    confidence_threshold=0.65,
)
print(out.get('final_report', {}).get('verdict'), out.get('final_report', {}).get('ml_confidence'))
print('RAG path used:', out.get('final_report', {}).get('rag_path_used'))
"
```

- **`confidence_threshold`**: if the model’s predicted-class probability is **below** this value, the graph runs **retrieve → verify (placeholder) → report**; otherwise it goes **straight to report**. Default constant: `DEFAULT_LOW_CONFIDENCE_THRESHOLD` in `src/agent/state.py`.

You can also call **`build_graph().invoke({"raw_text": "..."})`** for the same end-to-end run.

### D. One-shot local order (copy-paste)

From a fresh clone (after `venv` + `pip install -r requirements.txt`):

```bash
# Optional: train ML if you have dataset/ CSVs
python scripts/run_evaluation.py

# RAG index (downloads MiniLM on first run)
python scripts/build_rag_index.py

# Web UI
streamlit run app.py
```

Open **http://localhost:8501**. The Streamlit app uses the **ML pipeline** for live predictions; the **LangGraph + RAG** path is available programmatically via `invoke_credibility_agent` until a dedicated dashboard page is added.

---

## ML Pipeline

### Stage 1 — Text Preprocessing

Every document passes through the same deterministic `clean_text()` function at both training time and inference time, ensuring no preprocessing drift.

```
Raw string
  → lowercase
  → remove URLs (http / https / www)
  → remove @mentions
  → keep only [a-z] and whitespace
  → tokenize (split on whitespace)
  → remove NLTK English stopwords
  → remove tokens with length <= 2
  → WordNet lemmatization
  → rejoin tokens with space
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


| Hyperparameter | Value                              |
| -------------- | ---------------------------------- |
| `max_features` | 20,000 – 25,000 (model-dependent) |
| `ngram_range`  | (1, 2) — unigrams and bigrams     |
| `min_df`       | 2                                  |
| `max_df`       | 0.92                               |
| `sublinear_tf` | True                               |

### Stage 3 — Classification Models


| Model                   | Key Hyperparameters                                           | Notes                                        |
| ----------------------- | ------------------------------------------------------------- | -------------------------------------------- |
| **Logistic Regression** | `C=2.0`, `class_weight='balanced'`, `solver='lbfgs'`          | Interpretable; calibrated probabilities      |
| **Naive Bayes**         | `MultinomialNB(alpha=0.1)`                                    | Fast; works well with sparse TF-IDF matrices |
| **Random Forest**       | `n_estimators=200`, `max_depth=30`, `class_weight='balanced'` | Non-linear; robust to feature noise          |
| **SVM**                 | `LinearSVC(C=1.0)`, `class_weight='balanced'`                 | Strong linear decision boundary              |

The best model by F1 Score is saved as `model/pipeline.pkl` and used by the Streamlit application. All four models are compared in the notebook and in the **Model Comparison** dashboard page.

### Stage 4 — Evaluation

```python
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score

y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1', n_jobs=-1)
print(f"5-Fold CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
```

### Prediction Path (Inference)

```
User Input (raw text)
    |
    v
clean_text(input)                  # deterministic preprocessing
    |
    v
pipeline.predict([cleaned])        # TF-IDF transform → classifier forward pass
pipeline.predict_proba([cleaned])
    |
    v
Verdict: Fake / Real
Probability: P(Fake) = 0.XX
```

### Usage from Python

```python
import joblib
from src.features.preprocessing import clean_text

pipeline = joblib.load("model/pipeline.pkl")

text = "Your news headline or article body here..."
cleaned = clean_text(text)

label = pipeline.predict([cleaned])[0]         # 0 = Fake, 1 = Real
proba = pipeline.predict_proba([cleaned])[0]   # [P(Fake), P(Real)]

print(f"Verdict:          {'Fake' if label == 0 else 'Real'}")
print(f"Fake probability: {proba[0]:.2%}")
```

---

## Results

> Results are produced by running the notebook or `python scripts/run_evaluation.py`. The dashboard loads them from `model/evaluation_results.json`. No metrics are hardcoded.


| Metric                      | Logistic Regression | Naive Bayes  | Random Forest | SVM          |
| --------------------------- | ------------------- | ------------ | ------------- | ------------ |
| **Accuracy**                | High                | Competitive  | Competitive   | High         |
| **Precision / Recall / F1** | Strong              | Strong       | Strong        | Strong       |
| **ROC-AUC**                 | High                | Good         | Good          | High         |
| **5-Fold CV F1**            | See artifact        | See artifact | See artifact  | See artifact |

The best model by F1 is selected and persisted to `model/pipeline.pkl`. The notebook and **Model Comparison** dashboard page display ROC curves, Precision-Recall curves, confusion matrices, and a CV F1 bar chart for all four models.

### Metric Definitions


| Metric               | Definition                                                           | Why It Matters                            |
| -------------------- | -------------------------------------------------------------------- | ----------------------------------------- |
| **Precision (Fake)** | Of all articles flagged Fake, the fraction that are actually Fake    | Reduces false alarms on real news         |
| **Recall (Fake)**    | Of all actual Fake articles, the fraction the model correctly caught | Reduces missed misinformation             |
| **F1 Score**         | Harmonic mean of Precision and Recall                                | Primary optimization target               |
| **ROC-AUC**          | Ranking quality across all decision thresholds                       | Threshold-independent performance measure |

For misinformation detection, both Precision and Recall are critical: high Recall catches more fake news; high Precision avoids incorrectly flagging real news.

---

## Application

### Streamlit Dashboard — News Credibility Analyzer

[Live Demo](https://news-creditability.streamlit.app)

Run locally:

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

The application consists of five pages:


| Page                     | Description                                                                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **Overview**             | Key metrics (accuracy, F1, ROC-AUC, 5-fold CV F1), dataset summary, and attribution                               |
| **Dataset Intelligence** | Class distribution, text length distributions, top TF-IDF features by class                                       |
| **Model Comparison**     | ROC curves, Precision-Recall curves, confusion matrices, CV F1 distribution, feature importance for linear models |
| **Live Prediction Lab**  | Text input with sample articles, Analyze button → Fake/Real verdict with confidence score                        |
| **Architecture**         | Pipeline overview and repository structure map                                                                    |

The app reads `model/pipeline.pkl` (best model) and `model/evaluation_results.json` (metrics). If either file is missing, the relevant pages display a clear prompt to run the training script or notebook — no stale or hardcoded data is ever shown.

---

## Dashboard Metrics

The Streamlit dashboard displays dataset statistics and model metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrices, CV F1) from a single source of truth: `model/evaluation_results.json`.

**To generate the evaluation artifact:**

Option A — run `notebook/news_credibility.ipynb` from top to bottom (run from repo root so `dataset/` and `src/` are on the path). The notebook covers data loading, EDA, preprocessing, feature engineering, model training, evaluation, and artifact export.

Option B — run the evaluation script directly:

   ```bash
   python scripts/run_evaluation.py
   ```

This loads the data, trains all four models, evaluates on the stratified test set, selects the best by F1, and saves `model/evaluation_results.json` and `model/pipeline.pkl`.

---

## Deployment

This project is structured for deployment to **Streamlit Community Cloud** (free tier). Localhost-only demonstrations are not accepted per project requirements.

### Deploy to Streamlit Community Cloud

**Step 1 — Push to GitHub**

```bash
git add .
git commit -m "feat: milestone 1 complete"
git push origin main
```

> If `model/pipeline.pkl` exceeds 100 MB, use Git LFS:
>
> ```bash
> git lfs install && git lfs track "*.pkl"
> git add .gitattributes && git commit -m "chore: add git lfs"
> ```

**Step 2 — Connect to Streamlit**

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New App** and select your repository.
3. Set **Branch:** `main` and **Main file:** `app.py`.
4. Click **Deploy**.

**Step 3 — Verify**

Your live application will be available at the URL shown in the Streamlit Cloud dashboard.

### Environment Requirements


| Requirement | Value                                    |
| ----------- | ---------------------------------------- |
| Python      | 3.8 or higher                            |
| GPU         | Not required                             |
| RAM         | ~512 MB (model and vectorizer in memory) |
| Storage     | ~100–200 MB (pipeline.pkl)              |

### Dependencies (`requirements.txt`)

```
streamlit
scikit-learn
pandas
numpy
nltk
joblib
plotly
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Limitations


| Area                        | Limitation                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Language**                | Preprocessing uses English NLTK resources; the dataset is English-only                                                    |
| **Domain**                  | Trained on the Kaggle Fake and Real News corpus; may not generalize to other domains or topics                            |
| **Task Scope**              | Binary classification only (Fake/Real); no multi-label support for satire, misleading framing, or out-of-context articles |
| **Temporal Generalization** | Random train/test split; performance may be overstated if the news distribution shifts over time                          |
| **Model Ceiling**           | Classical TF-IDF with LR/SVM; not state-of-the-art compared to fine-tuned transformer models                              |
| **Probability Calibration** | `predict_proba` outputs are not formally calibrated (no Platt scaling or isotonic regression applied)                     |

---

## Future Work — Milestone 2 (Agentic AI)

Milestone 2 will extend this system into an **autonomous misinformation monitoring assistant** using LangGraph with the following capabilities:

**Agentic fact-checking** via a LangGraph workflow with RAG retrieval (Chroma/FAISS) against fact-checking corpora. **LLM reasoning** for structured credibility assessments with source cross-referencing and hallucination reduction. **Multi-label classification** expanding from binary labels to categories: satire, misleading, out-of-context, unverified. **Calibration** using Platt scaling for well-calibrated probability outputs. **REST API** for batch inference and integration into content moderation pipelines. **Transformer baseline** using fine-tuned IndicBERT or mBERT for comparison against the TF-IDF approach.

---

## Milestone 1 Deliverables Checklist

- [X]  Problem understanding and media use-case documented
- [X]  Input-output specification (`text → credibility label + probability`)
- [X]  System architecture diagram
- [X]  Working Streamlit application (**News Credibility Analyzer**) with multi-page UI
- [X]  Model performance evaluation report (Precision · Recall · F1 · ROC-AUC · CV F1)
- [X]  Multiple models trained and compared (Logistic Regression · Naive Bayes · Random Forest · SVM)
- [X]  Confusion matrices, ROC curves, and Precision-Recall curves
- [X]  TF-IDF feature interpretability (top fake/real indicative terms)
- [X]  Dataset attribution (Kaggle Fake and Real News) in app and README
- [X]  Publicly hosted application URL — [https://](https://)

---

<div align="center">

**Project 11 · Intelligent News Credibility Analysis and Agentic Misinformation Monitoring**

Milestone 1 — ML-Based Classification · Built with scikit-learn · Deployed on Streamlit

</div>
