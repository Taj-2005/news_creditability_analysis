"""
Streamlit UI for News Credibility Analyzer.

Loads the trained pipeline from model/pipeline.pkl and uses the same
preprocessing (clean_text) as training. Run with:
  streamlit run src/app/main.py
Or from repo root: streamlit run app.py (if app.py delegates here).
"""

import sys
from pathlib import Path

# Ensure project root is on path for "src" imports
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import streamlit as st

from src.features.preprocessing import clean_text


@st.cache_resource
def load_model():
    """Load the trained pipeline from model/pipeline.pkl."""
    import joblib

    model_path = repo_root / "model" / "pipeline.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first (see README)."
        )
    return joblib.load(model_path)


def main() -> None:
    st.set_page_config(
        page_title="News Credibility Analyzer",
        page_icon="📰",
        layout="centered",
    )
    st.title("📰 News Credibility Analyzer")
    st.markdown(
        """
        **Project 11 — Milestone 1** | BharatFakeNewsKosh Dataset  
        Paste a news article or statement below to check its credibility.
        """
    )
    st.divider()

    input_text = st.text_area(
        "📝 Enter News Article or Statement:",
        height=200,
        placeholder="Paste your news article text here...",
    )

    if st.button("🔍 Analyze Credibility", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("Analyzing..."):
                pipeline = load_model()
                cleaned = clean_text(input_text)
                prediction = pipeline.predict([cleaned])[0]
                proba = pipeline.predict_proba([cleaned])[0]
                fake_prob = float(proba[1])
                real_prob = float(proba[0])

            st.divider()

            if prediction == 1:
                st.error("🔴 Verdict: Likely FAKE / Misinformation")
            else:
                st.success("🟢 Verdict: Likely CREDIBLE / Real")

            col1, col2 = st.columns(2)
            col1.metric("Fake Probability", f"{fake_prob:.1%}")
            col2.metric("Real Probability", f"{real_prob:.1%}")

            st.markdown("**Credibility Risk Score:**")
            st.progress(fake_prob)

            with st.expander("ℹ️ How this works"):
                st.markdown(
                    """
                    This tool uses a **Logistic Regression** model trained on **26,000+ Indian news articles**
                    from the BharatFakeNewsKosh dataset.
                    Text is preprocessed (stopword removal, lemmatization) and vectorized using **TF-IDF (bigrams)**.
                    The model outputs a probability score — higher score = higher risk of misinformation.
                    \n⚠️ This is an AI tool. Always verify news with trusted sources.
                    """
                )

    st.divider()
    st.caption(
        "Project 11 | Intelligent News Credibility Analysis | Milestone 1"
    )


if __name__ == "__main__":
    main()
