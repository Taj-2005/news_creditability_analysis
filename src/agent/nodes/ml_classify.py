"""
ML classify node: run the existing credibility classifier on cleaned text.

Integration will call the same inference path as the Streamlit app
(e.g. ``run_prediction`` from ``src.app.core``) without modifying the
underlying ``pipeline.pkl`` or training code.
"""

from typing import Any, Dict

from src.agent.state import AgentState


def run_ml_classify_node(
    state: AgentState,
    *,
    pipeline: Any = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """
    Produce a partial state update with ML label and class scores.

    Args:
        state: Current graph state; must include ``cleaned_text`` when implemented.
        pipeline: Fitted sklearn ``Pipeline`` loaded via joblib; unused in scaffold.
        **_kwargs: Reserved for future options (e.g. batch mode).

    Returns:
        A mapping with keys such as ``ml_label``, ``ml_p_fake``, ``ml_p_real``.
        Currently returns an empty dict.
    """
    return {}


def describe_ml_classify_step() -> str:
    """
    Human-readable description of this node for docs or UI tooltips.

    Returns:
        Short summary string.
    """
    return "ML classify: TF-IDF + classifier verdict and scores."
