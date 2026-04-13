"""
Entry point for the News Credibility AI dashboard.

Run from repository root:
  streamlit run app.py

Multi-page dashboard: Home, Model comparison, Dataset insights, Live prediction, Architecture.
"""

from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.config.env_bootstrap import bootstrap_runtime_env

bootstrap_runtime_env()

from src.app.dashboard import run

if __name__ == "__main__":
    run()
