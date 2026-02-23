"""
Entry point for the News Credibility AI dashboard.

Run from repository root:
  streamlit run app.py

Multi-page dashboard: Home, Model comparison, Dataset insights, Live prediction, Architecture.
"""

from src.app.dashboard import run

if __name__ == "__main__":
    run()
