"""
Sidebar navigation: page order and one-line descriptions (Overview + dashboard).
"""

from __future__ import annotations

from typing import Dict, List

# Order must match the radio in ``dashboard.run``.
SIDEBAR_PAGE_ORDER: List[str] = [
    "Overview",
    "Dataset Intelligence",
    "Model Comparison",
    "Live Prediction Lab",
    "Deep Analysis",
    "Architecture",
]

SIDEBAR_PAGE_DESCRIPTIONS: Dict[str, str] = {
    "Overview": "Executive summary, KPIs, stack tour, and this app map.",
    "Dataset Intelligence": "Class balance, text-length profiles, and exploratory signals.",
    "Model Comparison": "ROC, PR, confusion matrices, and TF-IDF interpretability across models.",
    "Live Prediction Lab": "Fast Fake vs Real verdict from TF-IDF + saved best model (no RAG).",
    "Deep Analysis": "Full LangGraph run: FAISS/Chroma retrieval (similarity/MMR), verification JSON, structured report + feedback.",
    "Architecture": "Training vs runtime Mermaid, Milestone 2 pipeline diagram, repo map.",
}
