"""
Verify node: compare claims to retrieved evidence (placeholder, no LLM).

Future implementation will call an LLM with strict JSON schema; for now this
node records how many chunks were available and a static status string.
"""

from typing import Any, Dict

from src.agent.state import AgentState


def run_verify_node(state: AgentState, **_kwargs: Any) -> Dict[str, Any]:
    """
    Produce a placeholder ``verification`` dict for LangGraph state merging.

    Args:
        state: Expects ``retrieved_chunks`` (possibly empty) and ML fields.
        **_kwargs: Reserved for LLM client / prompts.

    Returns:
        Partial update with key ``verification`` only.
    """
    chunks = state.get("retrieved_chunks") or []
    rag_err = state.get("rag_error")

    verification: Dict[str, Any] = {
        "mode": "placeholder",
        "llm": False,
        "chunks_reviewed": len(chunks),
        "notes": (
            "LLM verification not enabled. Evidence snippets are attached in "
            "final_report for human review."
        ),
    }
    if rag_err:
        verification["rag_error"] = rag_err

    if chunks:
        verification["top_scores"] = [float(c.get("score", 0.0)) for c in chunks[:3]]
    else:
        verification["top_scores"] = []

    return {"verification": verification}
