<div align="center">

# 📰News Credibility Classification System

### Intelligent Misinformation Detection via Classical NLP & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-BharatFakeNewsKosh-6366F1?style=flat-square)](https://bharatfakenewskosh)

**Project 11 · Milestone 1 · AI/ML Systems**

[Overview](#-overview) · [Architecture](#-system-architecture) · [Quickstart](#-quickstart) · [Pipeline](#-ml-pipeline) · [Results](#-results) · [Deployment](#-deployment) · [Limitations](#-limitations)

</div>

---

## Overview

Misinformation spreads faster than manual fact-checking can scale. This project builds an **end-to-end, production-structured ML pipeline** that automatically classifies news articles as **Fake** or **Real** using classical NLP techniques — no LLMs, no GPU required.

Trained on **26,000+ fact-checked Indian news articles** from the [BharatFakeNewsKosh](https://www.kaggle.com/datasets/man2191989/bharatfakenewskosh) dataset, the system uses TF-IDF vectorization paired with Logistic Regression and Decision Tree classifiers, wrapped in a clean sklearn `Pipeline` and served through a Streamlit web application.

**This is Milestone 1** of a two-phase project. Milestone 2 extends the system into a LangGraph-based agentic AI assistant with RAG-powered fact-checking.

### Problem Statement


| Challenge                                 | Scale                         |
| ----------------------------------------- | ----------------------------- |
| News articles fact-checked daily in India | 100s per day                  |
| Manual verification bottleneck            | Hours per article             |
| Dataset size (BharatFakeNewsKosh)         | 26,232 articles               |
| Languages covered                         | 9 (Hindi, Tamil, English, …) |
| Class distribution                        | 60.6% Fake · 39.4% Real      |

---

## Project Team - Section A

| Team Member        | Role                         | Contribution                                                                                                                                                                                                 | GitHub Profile                           |
| ------------------ | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **Shaik Tajuddin** | Project Lead & GitHub Manager| Project leadership, repository creation & enhancement, roadmap planning, repo structure planning, notebook (ipynb) design, requirements & tech stack planning, PR reviews, version control & collaboration management | [@Taj-2005](https://github.com/Taj-2005) |
| **Nipun**          | Backend & Packaging Engineer | Folder architecture implementation, converting notebooks into modular Python files, building reusable ML pipeline, preparing codebase for Streamlit integration, project modularization                 | [@nipun1803](https://github.com/nipun1803)                         |
| **Hadole**         | Deployment & UI Engineer     | Streamlit application design, deployment architecture, model integration into UI, preparing deployable structure, hosting setup & deployment pipeline, usability flow                                   | [@omkar-hadole](https://github.com/omkar-hadole)                         |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                     INTELLIGENT NEWS CREDIBILITY ANALYSIS SYSTEM                          │
│                         Milestone 1 · Classical NLP + Machine Learning                    │
└──────────────────────────────────────────────────────────────────────────────────────────┘


                                    DATA SOURCE
                           ───────────────────────────
                             BharatFakeNewsKosh Dataset
                       (Eng_Trans_Statement + News_Body + Label)


╔══════════════════════════════════════ TRAINING PIPELINE ══════════════════════════════════════╗

        ┌────────────────────┐
        │   Data Loading     │
        │  Excel → DataFrame │
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Text Preparation   │
        │ Merge statement +  │
        │ article body       │
        └─────────┬──────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │              Preprocessing                │
        │------------------------------------------│
        │ • Lowercasing                            │
        │ • Remove URLs & punctuation              │
        │ • Stopword removal                       │
        │ • Tokenization                           │
        │ • Lemmatization (WordNet)                │
        └─────────┬────────────────────────────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │        Feature Engineering (TF-IDF)      │
        │------------------------------------------│
        │ • 15K max features                       │
        │ • Unigrams + Bigrams                     │
        │ • Sublinear TF scaling                   │
        └─────────┬────────────────────────────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │          Machine Learning Models         │
        │------------------------------------------│
        │ • Logistic Regression                    │
        │ • Decision Tree                          │
        │ • Probability Estimation                 │
        └─────────┬────────────────────────────────┘
                  ▼
        ┌────────────────────┐
        │ Model Evaluation   │
        │ Precision / Recall │
        │ F1 Score / Matrix  │
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Model Persistence  │
        │ Save pipeline.pkl  │
        └────────────────────┘

╚══════════════════════════════════════════════════════════════════════════════════════════════╝



╔══════════════════════════════════════ INFERENCE PIPELINE ═════════════════════════════════════╗

        User Input (Article Text / URL)
                    │
                    ▼
        ┌────────────────────┐
        │ Load Saved Model   │
        │ pipeline.pkl       │
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Clean Input Text   │
        │ Same preprocessing │
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Vectorization      │
        │ TF-IDF Transform   │
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ Prediction Engine  │
        │ predict_proba()    │
        └─────────┬──────────┘
                  ▼
        ┌────────────────────────────────────┐
        │ Credibility Assessment Output      │
        │------------------------------------│
        │ Fake / Real Verdict                │
        │ Confidence Score (0–1)             │
        │ Risk Level (High / Low)            │
        └────────────────────────────────────┘

╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Data Flow


| Stage         | Input                     | Output                                         | Module                          |
| ------------- | ------------------------- | ---------------------------------------------- | ------------------------------- |
| **Load**      | `bharatfakenewskosh.xlsx` | Raw DataFrame                                  | `src/data/loader.py`            |
| **Prepare**   | Raw DataFrame             | `combined_text`, `cleaned_text`, `label` (0/1) | `src/features/preprocessing.py` |
| **Split**     | X (text), y (label)       | Stratified 80/20 train/test sets               | `sklearn.model_selection`       |
| **Vectorize** | Text series               | Sparse TF-IDF matrix                           | `Pipeline` (fit on train only)  |
| **Train**     | TF-IDF matrix + labels    | Fitted Pipeline artifact                       | `src/models/pipelines.py`       |
| **Evaluate**  | Pipeline + test set       | Classification report, ROC-AUC, CM             | `src/evaluation/`               |
| **Serialize** | Fitted pipeline           | `model/pipeline.pkl`                           | `joblib`                        |
| **Serve**     | Raw user text             | Verdict + probability                          | `src/app/main.py`               |

> **Critical invariant:** The same `clean_text()` function and the same fitted pipeline (vectorizer + classifier) are used in both training and inference. No preprocessing drift.

---

## 📁 Repository Structure

```
GenAI/
├── README.md                          # ← You are here
├── ARCHITECTURE.md                    # Detailed system design docs
├── requirements.txt                   # Pinned dependencies
│
├── news_credibility.ipynb             # Main training + evaluation notebook
├── app.py                             # Streamlit app entry point
│
├── bharatfakenewskosh.xlsx            # Dataset (add locally; excluded from VCS if >50MB)
├── model/
│   └── pipeline.pkl                   # Serialized sklearn Pipeline (post-training)
│
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── loader.py                  # load_dataset(), prepare_text_column()
    ├── features/
    │   ├── __init__.py
    │   └── preprocessing.py           # clean_text() — shared between train + serve
    ├── models/
    │   ├── __init__.py
    │   └── pipelines.py               # build_lr_pipeline(), build_dt_pipeline()
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py                 # print_report(), cross_validate()
    │   └── visualization.py           # plot_confusion_matrix(), plot_roc()
    ├── utils/
    │   └── __init__.py
    └── app/
        ├── __init__.py
        └── main.py                    # Streamlit UI (alternate entry point)
```

---

## Quickstart

### 1. Clone & enter the repo

```bash
git clone https://github.com/Taj-2005/news_creditability_analysis
cd GenAI
```

### 2. Set up environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the dataset

Place `bharatfakenewskosh.xlsx` in the project root. The file must contain:


| Column                | Type | Description                               |
| --------------------- | ---- | ----------------------------------------- |
| `Eng_Trans_Statement` | str  | English translation of the claim/headline |
| `Eng_Trans_News_Body` | str  | English translation of the news body      |
| `Label`               | bool | `True` = Fake, `False` = Real             |

### 4. Train the model

```bash
jupyter notebook news_credibility.ipynb
# Run all cells — model saves to model/pipeline.pkl
```

### 5. Launch the app

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 🔬 ML Pipeline

### Stage 1 — Text Preprocessing

Every document goes through the same deterministic preprocessing function:

```
Raw string
  → lowercase
  → remove URLs (http/https/www)
  → remove @mentions
  → keep only [a-z] and whitespace
  → tokenize (split on whitespace)
  → remove NLTK English stopwords
  → remove tokens with len ≤ 2
  → WordNet lemmatization
  → rejoin with space
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


| Hyperparameter | Logistic Regression | Decision Tree |
| -------------- | ------------------- | ------------- |
| `max_features` | 15,000              | 10,000        |
| `ngram_range`  | (1, 2)              | (1, 2)        |
| `min_df`       | 3                   | 3             |
| `max_df`       | 0.90                | 0.90          |
| `sublinear_tf` | True                | True          |

### Stage 3 — Classification


| Model                   | Key Hyperparameters                                               | Rationale                                                                     |
| ----------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Logistic Regression** | `C=1.0`, `class_weight='balanced'`, `solver='lbfgs'`              | Interpretable coefficients; calibrated probabilities; handles class imbalance |
| **Decision Tree**       | `max_depth=25`, `min_samples_split=10`, `class_weight='balanced'` | Non-linear splits; depth-limited to reduce overfitting                        |

All stages are encapsulated in a single sklearn `Pipeline` object so the vectorizer is fit **only on training data** and the same vocabulary is used at inference time.

### Stage 4 — Evaluation

```python
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score

y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1', n_jobs=-1)
print(f"5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

## Results

> Results shown below are expected based on dataset characteristics. Exact numbers populate after running the notebook.


| Metric               | Logistic Regression | Decision Tree |
| -------------------- | ------------------- | ------------- |
| **Accuracy**         | ~0.87               | ~0.80         |
| **Precision (Fake)** | ~0.88               | ~0.81         |
| **Recall (Fake)**    | ~0.90               | ~0.84         |
| **F1 Score (Fake)**  | ~0.89               | ~0.82         |
| **ROC-AUC**          | ~0.93               | ~0.80         |
| **5-Fold CV F1**     | ~0.88 ± 0.01       | ~0.81 ± 0.02 |

**Why Logistic Regression outperforms Decision Tree on text data:**
TF-IDF produces high-dimensional sparse features where linear separability is strong. LR exploits this directly via a linear decision boundary, while Decision Trees must construct many splits to approximate the same boundary, leading to overfitting.

### Metric Definitions


| Metric               | Definition                                                    | Priority                      |
| -------------------- | ------------------------------------------------------------- | ----------------------------- |
| **Precision (Fake)** | Of all articles flagged Fake, fraction that are actually Fake | Reduces false alarms          |
| **Recall (Fake)**    | Of all actual Fake articles, fraction the model caught        | Reduces missed misinformation |
| **F1 Score**         | Harmonic mean of Precision and Recall                         | Primary optimization target   |
| **ROC-AUC**          | Ranking quality across all decision thresholds                | Threshold-independent         |

For misinformation detection, **both Precision and Recall matter**: high Recall catches more fake news; high Precision avoids flagging real news as fake.

---

## Application

### Prediction Path (Inference)

```
User Input (raw text)
    ↓
clean_text(input)               # deterministic preprocessing
    ↓
pipeline.predict([cleaned])     # TF-IDF transform → classifier forward pass
pipeline.predict_proba([cleaned])
    ↓
Verdict: Fake / Real
Probability bar: P(Fake) = 0.XX
```

### Usage from Python

```python
import joblib
from src.features.preprocessing import clean_text

pipeline = joblib.load("model/pipeline.pkl")

text = "Your news headline or article body here..."
cleaned = clean_text(text)

label = pipeline.predict([cleaned])[0]         # 0 = Real, 1 = Fake
proba = pipeline.predict_proba([cleaned])[0]   # [P(Real), P(Fake)]

print(f"Verdict:          {'Fake' if label else 'Real'}")
print(f"Fake probability: {proba[1]:.2%}")
```

---

## Deployment

This project is structured for deployment to Streamlit Community Cloud (free tier). **Localhost-only demonstrations are not accepted per project requirements.**

### Deploy to Streamlit Community Cloud

**Step 1 — Push to GitHub**

```bash
git add .
git commit -m "feat: milestone 1 complete"
git push origin main
```

> If `model/pipeline.pkl` exceeds 100MB, use Git LFS:
>
> ```bash
> git lfs install && git lfs track "*.pkl"
> git add .gitattributes && git commit -m "chore: add git lfs"
> ```

**Step 2 — Connect to Streamlit**

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New App** → select your repository
3. Set **Branch:** `main` · **Main file:** `app.py`
4. Click **Deploy**

**Step 3 — Verify**

Your app will be live at `https://<username>-<repo>-app-<hash>.streamlit.app`

### Environment Requirements

```
Python:  3.8+
GPU:     Not required
RAM:     ~512MB (model + vectorizer in memory)
Storage: ~100–200MB (pipeline.pkl)
```

### `requirements.txt`

```
streamlit>=1.28
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
nltk>=3.8
joblib>=1.3
openpyxl>=3.1
```

---

## Limitations


| Area            | Limitation                                                                                   |
| --------------- | -------------------------------------------------------------------------------------------- |
| **Language**    | Preprocessing uses English NLTK resources; relies on English-translated columns              |
| **Domain**      | Trained on Indian fact-checked news; may not generalize to other regions or domains          |
| **Task**        | Binary classification only (Fake/Real); no multi-label (satire, misleading, out-of-context)  |
| **Temporal**    | Random train/test split; performance may be overstated if distribution shifts over time      |
| **Model**       | Classical TF-IDF + LR/DT; not state-of-the-art versus fine-tuned transformer classifiers     |
| **Calibration** | `predict_proba` outputs are not formally calibrated (no Platt scaling / isotonic regression) |

---

## Future Work — Milestone 2 (Agentic AI)

Milestone 2 extends this system into an **autonomous misinformation monitoring assistant** using LangGraph:

- **Agentic fact-checking** — LangGraph workflow with RAG retrieval (Chroma/FAISS) against fact-checking corpora
- **LLM reasoning** — structured credibility assessments with source cross-referencing and hallucination reduction
- **Multi-label classification** — expand from binary to categories: satire, misleading, out-of-context, unverified
- **Calibration** — Platt scaling for well-calibrated probability outputs
- **REST API** — batch inference endpoint for integration into content moderation pipelines
- **Transformer baseline** — fine-tuned IndicBERT or mBERT for comparison against TF-IDF + LR

---

## Milestone 1 Deliverables Checklist

- [X]  Problem understanding & media use-case documented
- [X]  Input–output specification (`text → credibility label + probability`)
- [X]  System architecture diagram
- [X]  Working Streamlit application with UI
- [X]  Model performance evaluation report (Precision · Recall · F1 · ROC-AUC · CV)
- [X]  Two models trained and compared (Logistic Regression · Decision Tree)
- [X]  Confusion matrices and ROC curves
- [X]  TF-IDF feature interpretability (top fake/real words)
- [ ]  Publicly hosted application URL _(post-deployment)_

---

<div align="center">

**Project 11 · Intelligent News Credibility Analysis & Agentic Misinformation Monitoring**
Milestone 1 — ML-Based Classification · Built with scikit-learn · Deployed on Streamlit

</div>
