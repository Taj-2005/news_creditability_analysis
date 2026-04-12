"""
Agent orchestration package (Milestone 2).

LangGraph workflow composes normalize → ML → optional RAG → report without
modifying training code under ``src.features`` / ``src.models``.
"""

from src.agent.graph import build_graph, invoke_credibility_agent
from src.agent.llm_service import generate, is_configured
from src.agent.state import AgentState, DEFAULT_LOW_CONFIDENCE_THRESHOLD

__all__ = [
    "AgentState",
    "DEFAULT_LOW_CONFIDENCE_THRESHOLD",
    "build_graph",
    "generate",
    "invoke_credibility_agent",
    "is_configured",
]
