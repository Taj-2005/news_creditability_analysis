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
        rag_backend: Retrieval backend ("faiss" | "chroma").
        rag_search_type: Retrieval mode ("similarity" | "mmr").
        rag_fetch_k: Candidate pool size for MMR.
        rag_lambda_mult: MMR tradeoff (0..1).
        rag_error: Set when the retriever skips or fails (index missing, etc.).
        verification: Verifier output; always includes ``supported``, ``contradicted``,
            ``unknown`` (each a ``list[str]``), plus ``mode``, ``llm``, ``chunks_reviewed``.
        final_report: UI payload with ``summary``, ``risk_factors``, ``fact_checks``,
            ``verdict``, ``confidence``, ``sources``, ``credibility_score`` (High | Low),
            ``pattern_detection_summary``, ``disclaimer`` (see ``src.agent.ui_report``).
        report_attempt: Integer attempt count for the report/validation loop.
        validation_passed: Whether output validation succeeded.
        validation_errors: Short strings describing failed validations.
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
    rag_backend: str
    rag_search_type: str
    rag_fetch_k: int
    rag_lambda_mult: float
    rag_error: Optional[str]
    verification: Dict[str, Any]
    final_report: Dict[str, Any]
    report_attempt: int
    validation_passed: bool
    validation_errors: List[str]
    error: Optional[str]


def empty_state() -> AgentState:
    """
    Return a minimal empty state for testing graph assembly.

    Returns:
        An ``AgentState`` with no keys set (all fields optional).
    """
    return {}
