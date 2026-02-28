"""
Production-grade design system: pure white SaaS theme, cards, typography.
Injected once via st.markdown(unsafe_allow_html=True).
"""

APP_CSS = """
<style>
  /* Base app */
  .stApp { background-color: #ffffff; }
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
  
  /* Typography — light theme: no black; use gray-700/600 for contrast */
  h1, h2, h3 { color: #374151; font-weight: 600; letter-spacing: -0.02em; }
  h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }
  h2 { font-size: 1.35rem; margin-top: 2rem; margin-bottom: 0.75rem; }
  h3 { font-size: 1.1rem; margin-top: 1.5rem; }
  p, .stMarkdown { color: #4b5563; line-height: 1.6; }
  hr { margin: 2rem 0; border: none; border-top: 1px solid #e5e7eb; }
  
  /* Sidebar — light only */
  [data-testid="stSidebar"] { background: #f9fafb; }
  [data-testid="stSidebar"] .stMarkdown { color: #6b7280; }
  [data-testid="stSidebar"] hr { border-color: #e5e7eb; }
  
  /* Cards */
  [data-testid="stVerticalBlock"] > div { border-radius: 12px; }
  
  /* Metric cards — light shadows (gray, not black) */
  [data-testid="stMetric"] {
    background: #ffffff;
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(107, 114, 128, 0.1);
    border: 1px solid #e5e7eb;
  }
  [data-testid="stMetricLabel"] { color: #6b7280; font-size: 0.8rem; font-weight: 500; }
  [data-testid="stMetricValue"] { color: #374151; font-size: 1.5rem; font-weight: 600; }
  
  /* Buttons */
  .stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: box-shadow 0.2s, background-color 0.2s;
  }
  .stButton > button[kind="primary"] { background-color: #2563eb; }
  .stButton > button[kind="primary"]:hover { background-color: #1d4ed8; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3); }
  
  /* Text area */
  .stTextArea textarea { border-radius: 10px; border: 1px solid #e5e7eb; }
  .stTextArea textarea:focus { border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15); }
  
  /* Expander */
  [data-testid="stExpander"] { border: 1px solid #e5e7eb; border-radius: 10px; background: #ffffff; }
  [data-testid="stExpander"] summary, [data-testid="stExpander"] label { color: #374151 !important; }
  
  /* DataFrames — light shadow */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(107, 114, 128, 0.1); }
  
  /* Hide Streamlit branding in footer for cleaner look */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  
  /* Force light text everywhere — no dark/black */
  .stCaption, [data-testid="stCaptionContainer"] { color: #6b7280 !important; }
  label { color: #374151 !important; }
  .stRadio label, .stCheckbox label { color: #4b5563 !important; }
  input, textarea { color: #374151 !important; background-color: #ffffff !important; }
</style>
"""

# Card wrapper class for use in custom HTML
CARD_CLASS = "credibility-card"


def inject_app_css():
    """Inject the main app CSS. Call once in dashboard run()."""
    import streamlit as st
    st.markdown(APP_CSS, unsafe_allow_html=True)
