"""
Agent orchestration package (Milestone 2).

LangGraph workflow composes normalize → ML → optional RAG → report without
modifying training code under ``src.features`` / ``src.models``.
"""

from src.agent.graph import build_graph, invoke_credibility_agent
from src.agent.llm_service import generate, is_configured
from src.agent.state import AgentState, DEFAULT_LOW_CONFIDENCE_THRESHOLD
from src.agent.ui_report import build_ui_final_report

__all__ = [
    "AgentState",
    "DEFAULT_LOW_CONFIDENCE_THRESHOLD",
    "build_graph",
    "build_ui_final_report",
    "generate",
    "invoke_credibility_agent",
    "is_configured",
]
