"""
LangGraph workflow: normalize → ML → (optional plan_queries + RAG + verify) → report → validate_report.

Low ML confidence: plan_queries (LLM) → retrieve (FAISS/Chroma; similarity/MMR) → verify (LLM) → report.
High confidence: report only (still may use an LLM for narrative when configured).

``validate_report`` can trigger **one** additional ``report`` attempt on schema failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Optional, Tuple, cast

from langgraph.graph import END, START, StateGraph

from src.agent.nodes.ml_classify import run_ml_classify_node
from src.agent.nodes.normalize import run_normalize_node
from src.agent.nodes.plan_queries import run_plan_queries_node
from src.agent.nodes.report import run_report_node
from src.agent.nodes.retrieve import run_retrieve_node
from src.agent.nodes.validate_report import run_validate_report_node
from src.agent.nodes.verify import run_verify_node
from src.agent.state import DEFAULT_LOW_CONFIDENCE_THRESHOLD, AgentState


def _route_after_ml(
    state: AgentState,
    *,
    threshold: float,
) -> Literal["plan_queries", "report"]:
    """
    Branch after ``ml_classify``: low confidence → query planning + RAG path.

    On ML or upstream errors, skip retrieval and still produce a report.
    """
    if state.get("error"):
        return "report"
    conf = state.get("ml_confidence")
    if conf is None:
        return "report"
    if float(conf) < float(threshold):
        return "plan_queries"
    return "report"


def build_graph(
    *,
    pipeline: Any = None,
    store_dir: Optional[Path] = None,
    confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    top_k: int = 5,
    rag_backend: str = "faiss",
    rag_search_type: str = "similarity",
    rag_fetch_k: int = 20,
    rag_lambda_mult: float = 0.6,
) -> Any:
    """
    Compile the credibility agent LangGraph.

    Args:
        pipeline: Optional sklearn pipeline for tests; default loads via ``core.load_model``.
        store_dir: RAG index directory (default ``<repo>/data/rag``).
        confidence_threshold: Below this ``ml_confidence``, run plan_queries → retrieve → verify.
        top_k: RAG hits when the retrieval path runs.

    Returns:
        A compiled LangGraph runnable (``invoke`` / ``ainvoke``).
    """
    sd = Path(store_dir) if store_dir is not None else None

    def _normalize(state: AgentState) -> dict:
        return run_normalize_node(state)

    def _ml(state: AgentState) -> dict:
        return run_ml_classify_node(state, pipeline=pipeline)

    def _plan(state: AgentState) -> dict:
        return run_plan_queries_node(state)

    def _retrieve(state: AgentState) -> dict:
        # Per-run overrides can be set in state by the UI; graph args are defaults.
        backend = str(state.get("rag_backend") or rag_backend)
        search_type = str(state.get("rag_search_type") or rag_search_type)
        fetch_k = int(state.get("rag_fetch_k") or rag_fetch_k)
        lam = float(state.get("rag_lambda_mult") or rag_lambda_mult)
        return run_retrieve_node(
            state,
            store_dir=sd,
            top_k=top_k,
            backend=backend,
            search_type=search_type,
            fetch_k=fetch_k,
            lambda_mult=lam,
        )

    def _verify(state: AgentState) -> dict:
        return run_verify_node(state)

    def _report(state: AgentState) -> dict:
        return run_report_node(state)

    def _validate(state: AgentState) -> dict:
        return run_validate_report_node(state)

    def _router(state: AgentState) -> Literal["plan_queries", "report"]:
        return _route_after_ml(state, threshold=confidence_threshold)

    def _route_after_validate(state: AgentState) -> Literal["end", "retry_report"]:
        if state.get("validation_passed"):
            return "end"
        # Bound retries to 1 additional attempt.
        attempt = int(state.get("report_attempt") or 0)
        if attempt >= 1:
            return "end"
        return "retry_report"

    graph = StateGraph(AgentState)
    graph.add_node("normalize", _normalize)
    graph.add_node("ml_classify", _ml)
    graph.add_node("plan_queries", _plan)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("verify", _verify)
    graph.add_node("report", _report)
    graph.add_node("validate_report", _validate)

    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "ml_classify")
    graph.add_conditional_edges(
        "ml_classify",
        _router,
        {"plan_queries": "plan_queries", "report": "report"},
    )
    graph.add_edge("plan_queries", "retrieve")
    graph.add_edge("retrieve", "verify")
    graph.add_edge("verify", "report")
    graph.add_edge("report", "validate_report")
    graph.add_conditional_edges(
        "validate_report",
        _route_after_validate,
        {"end": END, "retry_report": "report"},
    )

    return graph.compile()


def get_entry_node() -> str:
    """First business node after ``START``."""
    return "normalize"


def iter_credibility_agent(
    raw_text: str,
    *,
    pipeline: Any = None,
    store_dir: Optional[Path] = None,
    confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    top_k: int = 5,
    rag_backend: str = "faiss",
    rag_search_type: str = "similarity",
    rag_fetch_k: int = 20,
    rag_lambda_mult: float = 0.6,
) -> Iterator[Tuple[str, AgentState]]:
    """
    Stream graph execution: after each node, yield ``(node_name, merged_state)``.

    Uses LangGraph ``stream_mode="updates"`` so UIs can show live progress
    (e.g. Deep Analysis timeline).
    """
    graph = build_graph(
        pipeline=pipeline,
        store_dir=store_dir,
        confidence_threshold=confidence_threshold,
        top_k=top_k,
        rag_backend=rag_backend,
        rag_search_type=rag_search_type,
        rag_fetch_k=rag_fetch_k,
        rag_lambda_mult=rag_lambda_mult,
    )
    seed: Dict[str, Any] = {
        "raw_text": (raw_text or "").strip(),
        "rag_backend": rag_backend,
        "rag_search_type": rag_search_type,
        "rag_fetch_k": int(rag_fetch_k),
        "rag_lambda_mult": float(rag_lambda_mult),
    }
    merged: Dict[str, Any] = dict(seed)
    for chunk in graph.stream(seed, stream_mode="updates"):
        for node_name, update in chunk.items():
            merged = {**merged, **update}
            yield str(node_name), cast(AgentState, dict(merged))


def invoke_credibility_agent(
    raw_text: str,
    *,
    pipeline: Any = None,
    store_dir: Optional[Path] = None,
    confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    top_k: int = 5,
    rag_backend: str = "faiss",
    rag_search_type: str = "similarity",
    rag_fetch_k: int = 20,
    rag_lambda_mult: float = 0.6,
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
    last: Optional[AgentState] = None
    for _, state in iter_credibility_agent(
        raw_text,
        pipeline=pipeline,
        store_dir=store_dir,
        confidence_threshold=confidence_threshold,
        top_k=top_k,
        rag_backend=rag_backend,
        rag_search_type=rag_search_type,
        rag_fetch_k=rag_fetch_k,
        rag_lambda_mult=rag_lambda_mult,
    ):
        last = state
    return last or cast(AgentState, {"raw_text": (raw_text or "").strip()})


def run_graph_stub(state: AgentState) -> AgentState:
    """
    Run the compiled graph when ``raw_text`` is present; otherwise no-op.

    Prefer ``invoke_credibility_agent`` or ``build_graph().invoke`` for new code.
    """
    raw = (state.get("raw_text") or "").strip()
    if not raw:
        return state
    return invoke_credibility_agent(raw)
