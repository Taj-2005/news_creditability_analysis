"""
Typed agent state for LangGraph workflows.

Nodes return partial dicts that LangGraph merges into this state between steps.
"""

from typing import Any, Dict, List, Optional, TypedDict

# ML confidence below this routes through RAG → verify → report.
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.65


class AgentState(TypedDict, total=False):
    """
    State container passed between graph nodes.

    Attributes (all optional until populated by the graph):
        raw_text: Original user input.
        cleaned_text: Normalized text after preprocessing.
        ml_label: Binary classifier output (0=Fake, 1=Real) when available.
        ml_confidence: Predicted-class probability (same as ml_p_real if label==1, else ml_p_fake).
        ml_p_fake: Estimated probability or score for Fake class.
        ml_p_real: Estimated probability or score for Real class.
        queries: Search strings produced by ``plan_queries`` (low-confidence path).
        llm_query_error: Populated when Groq query planning fails (fallback queries used).
        retrieved_chunks: Top-k RAG hits; each item includes text, score, id, metadata.
        rag_error: Set when the retriever skips or fails (index missing, etc.).
        verification: Verifier output (LLM analysis or fallback metadata).
        final_report: Structured payload for UI / export (includes optional ``llm_summary``).
        error: Normalize / ML failure message; graph still reaches ``report``.
    """

    raw_text: str
    cleaned_text: str
    ml_label: int
    ml_confidence: float
    ml_p_fake: float
    ml_p_real: float
    queries: List[str]
    llm_query_error: Optional[str]
    retrieved_chunks: List[Dict[str, Any]]
    rag_error: Optional[str]
    verification: Dict[str, Any]
    final_report: Dict[str, Any]
    error: Optional[str]


def empty_state() -> AgentState:
    """
    Return a minimal empty state for testing graph assembly.

    Returns:
        An ``AgentState`` with no keys set (all fields optional).
    """
    return {}
