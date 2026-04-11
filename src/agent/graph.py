"""
LangGraph workflow: normalize → ML → (optional RAG) → report.

Low ML confidence routes through retrieve → verify (placeholder) → report;
high confidence goes directly to report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from langgraph.graph import END, START, StateGraph

from src.agent.nodes.ml_classify import run_ml_classify_node
from src.agent.nodes.normalize import run_normalize_node
from src.agent.nodes.report import run_report_node
from src.agent.nodes.retrieve import run_retrieve_node
from src.agent.nodes.verify import run_verify_node
from src.agent.state import DEFAULT_LOW_CONFIDENCE_THRESHOLD, AgentState


def _route_after_ml(
    state: AgentState,
    *,
    threshold: float,
) -> Literal["retrieve", "report"]:
    """
    Branch after ``ml_classify``: low confidence → RAG path; else → report only.

    On ML or upstream errors, skip retrieval and still produce a report.
    """
    if state.get("error"):
        return "report"
    conf = state.get("ml_confidence")
    if conf is None:
        return "report"
    if float(conf) < float(threshold):
        return "retrieve"
    return "report"


def build_graph(
    *,
    pipeline: Any = None,
    store_dir: Optional[Path] = None,
    confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    top_k: int = 5,
) -> Any:
    """
    Compile the credibility agent LangGraph.

    Args:
        pipeline: Optional sklearn pipeline for tests; default loads via ``core.load_model``.
        store_dir: RAG index directory (default ``<repo>/data/rag``).
        confidence_threshold: Below this ``ml_confidence``, run retrieve → verify.
        top_k: RAG hits when the retrieval path runs.

    Returns:
        A compiled LangGraph runnable (``invoke`` / ``ainvoke``).
    """
    sd = Path(store_dir) if store_dir is not None else None

    def _normalize(state: AgentState) -> dict:
        return run_normalize_node(state)

    def _ml(state: AgentState) -> dict:
        return run_ml_classify_node(state, pipeline=pipeline)

    def _retrieve(state: AgentState) -> dict:
        return run_retrieve_node(state, store_dir=sd, top_k=top_k)

    def _verify(state: AgentState) -> dict:
        return run_verify_node(state)

    def _report(state: AgentState) -> dict:
        return run_report_node(state)

    def _router(state: AgentState) -> Literal["retrieve", "report"]:
        return _route_after_ml(state, threshold=confidence_threshold)

    graph = StateGraph(AgentState)
    graph.add_node("normalize", _normalize)
    graph.add_node("ml_classify", _ml)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("verify", _verify)
    graph.add_node("report", _report)

    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "ml_classify")
    graph.add_conditional_edges(
        "ml_classify",
        _router,
        {"retrieve": "retrieve", "report": "report"},
    )
    graph.add_edge("retrieve", "verify")
    graph.add_edge("verify", "report")
    graph.add_edge("report", END)

    return graph.compile()


def get_entry_node() -> str:
    """First business node after ``START``."""
    return "normalize"


def invoke_credibility_agent(
    raw_text: str,
    *,
    pipeline: Any = None,
    store_dir: Optional[Path] = None,
    confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    top_k: int = 5,
) -> AgentState:
    """
    Run the full graph from raw user text (convenience for CLI / Streamlit).

    Args:
        raw_text: News text to analyze.
        pipeline: Optional injected ML pipeline.
        store_dir: RAG index directory.
        confidence_threshold: ML confidence threshold for RAG branch.
        top_k: Number of RAG chunks when the retrieval path runs.

    Returns:
        Final merged ``AgentState`` including ``final_report``.
    """
    graph = build_graph(
        pipeline=pipeline,
        store_dir=store_dir,
        confidence_threshold=confidence_threshold,
        top_k=top_k,
    )
    return graph.invoke({"raw_text": raw_text})


def run_graph_stub(state: AgentState) -> AgentState:
    """
    Run the compiled graph when ``raw_text`` is present; otherwise no-op.

    Prefer ``invoke_credibility_agent`` or ``build_graph().invoke`` for new code.
    """
    raw = (state.get("raw_text") or "").strip()
    if not raw:
        return state
    return invoke_credibility_agent(raw)
