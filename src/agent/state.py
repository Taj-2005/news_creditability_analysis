"""
Typed agent state for LangGraph-style workflows.

Defines the fields that nodes will read and update. Values are placeholders
until Milestone 2 wiring is implemented.
"""

from typing import Any, Dict, List, Optional, TypedDict


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
        queries: Search queries for retrieval (future RAG).
        retrieved_chunks: Evidence snippets from the vector store.
        verification: Structured verifier output (claims vs evidence).
        final_report: Structured or rendered report for the UI.
        error: Non-empty if a node failed; graph may short-circuit.
    """

    raw_text: str
    cleaned_text: str
    ml_label: int
    ml_confidence: float
    ml_p_fake: float
    ml_p_real: float
    queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]
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
