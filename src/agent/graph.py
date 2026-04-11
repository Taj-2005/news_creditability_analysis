"""
Graph builder for the agent workflow.

Responsible for registering nodes and edges (e.g. LangGraph ``StateGraph``).
Implementation is deferred; this module defines the public API surface only.
"""

from typing import Any, Callable, Optional

from src.agent.state import AgentState


def build_graph(
    *,
    pipeline: Any = None,
    retriever: Any = None,
    llm: Any = None,
) -> Optional[Any]:
    """
    Construct a compiled agent graph.

    Future implementation will accept the trained sklearn ``pipeline``,
    a RAG retriever, and an LLM client. Nodes will be wired according to
    the Milestone 2 design (normalize → ml_classify → … → report).

    Args:
        pipeline: Serialized TF-IDF + classifier (``joblib``), not used yet.
        retriever: Vector store interface for RAG, not used yet.
        llm: Language model client for planning / verification, not used yet.

    Returns:
        A compiled graph object when implemented; currently ``None``.
    """
    return None


def get_entry_node() -> Optional[str]:
    """
    Return the name of the graph entry node.

    Returns:
        Identifier string for the first node after ``START``, or ``None``
        until the graph is implemented.
    """
    return None


def run_graph_stub(state: AgentState) -> AgentState:
    """
    Pass-through placeholder for end-to-end graph execution.

    Args:
        state: Current agent state.

    Returns:
        The same state unchanged (placeholder behavior).
    """
    return state
