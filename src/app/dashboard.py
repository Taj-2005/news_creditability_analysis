"""
News Credibility AI — production-grade multi-page dashboard.
Pure white SaaS theme, wide layout, sidebar navigation, Plotly-only charts.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import streamlit as st

from src.app.components.styles import inject_app_css
from src.app.core import MODEL_ALGORITHM, PAGE_TITLE
from src.app.pages import (
    architecture,
    dataset_insights,
    home,
    live_prediction,
    model_compare,
)

# Page map: label -> (short description, render_fn)
PAGES = {
    "Overview": ("Executive summary & KPIs", home.render),
    "Dataset Intelligence": ("Class distribution & stats", dataset_insights.render),
    "Model Comparison": ("LR vs DT metrics & charts", model_compare.render),
    "Live Prediction Lab": ("Try the model", live_prediction.render),
    "Architecture": ("Pipeline & repo mapping", architecture.render),
}

APP_VERSION = "1.0.0"


def run():
    st.set_page_config(
        layout="wide",
        page_title=PAGE_TITLE,
        page_icon="📰",
        initial_sidebar_state="expanded",
    )
    inject_app_css()

    # Sidebar: minimal branding, nav, version, model
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 0.5rem;">
                <p style="font-size: 1rem; font-weight: 600; color: #374151; margin: 0;">News Credibility AI</p>
                <p style="font-size: 0.75rem; color: #6b7280; margin: 0.25rem 0 0 0;">Misinformation detection</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        page = st.radio(
            "Navigate",
            list(PAGES.keys()),
            format_func=lambda x: f"{x}",
            label_visibility="collapsed",
            key="nav_radio",
        )
        st.markdown("---")
        st.caption(f"Model · {MODEL_ALGORITHM}")
        st.caption(f"v{APP_VERSION}")

    # Render selected page
    PAGES[page][1]()


if __name__ == "__main__":
    run()
