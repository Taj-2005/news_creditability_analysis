"""
ML classify node: run the existing credibility classifier on cleaned text.

Uses ``load_model`` and ``run_prediction`` from ``src.app.core`` so inference
matches the Streamlit app with no duplicated classifier or TF-IDF logic.
"""

from typing import Any, Dict, Optional, Tuple

from src.agent.state import AgentState


def classify_cleaned_text(
    cleaned_text: str,
    pipeline: Any = None,
) -> Tuple[int, float, float, float]:
    """
    Run the shared inference path on text that has already been normalized.

    ``run_prediction`` internally applies ``clean_text`` again for the same
    train/serve contract as the dashboard; pass the same string you store in
    ``AgentState["cleaned_text"]`` (typically produced by the normalize node).

    Args:
        cleaned_text: Normalized document text (non-empty after strip).
        pipeline: Optional pre-loaded sklearn ``Pipeline``. If ``None``,
            ``load_model()`` from ``src.app.core`` is used.

    Returns:
        ``(ml_label, ml_confidence, ml_p_fake, ml_p_real)`` where
        ``ml_label`` is 0 (Fake) or 1 (Real), and ``ml_confidence`` is the
        model's probability for that class (from ``predict_proba`` or the
        SVM pseudo-probability path in ``run_prediction``).

    Raises:
        ValueError: If ``cleaned_text`` is empty after stripping.
        FileNotFoundError: If no ``model/pipeline.pkl`` is available.
    """
    from src.app.core import load_model, run_prediction

    text = (cleaned_text or "").strip()
    if not text:
        raise ValueError("cleaned_text must be non-empty after strip().")

    pipe = pipeline if pipeline is not None else load_model()
    label, p_fake, p_real = run_prediction(pipe, text)
    confidence = float(p_real) if int(label) == 1 else float(p_fake)
    return int(label), confidence, float(p_fake), float(p_real)


def run_ml_classify_node(
    state: AgentState,
    *,
    pipeline: Any = None,
    cleaned_text: Optional[str] = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """
    LangGraph-friendly partial state update from the credibility classifier.

    Reads ``cleaned_text`` from ``state`` or from the ``cleaned_text`` keyword
    (keyword wins for tests). On failure, sets ``error`` and omits ML fields.

    Args:
        state: Graph state; should contain ``cleaned_text`` after normalize.
        pipeline: Optional injected pipeline (e.g. tests); default loads via core.
        cleaned_text: Optional override; if provided, used instead of state.
        **_kwargs: Reserved for future flags.

    Returns:
        A dict mergeable into ``AgentState``, with keys ``ml_label``,
        ``ml_confidence``, ``ml_p_fake``, ``ml_p_real`` on success, or
        ``{"error": "<message>"}`` on failure.
    """
    text = (cleaned_text if cleaned_text is not None else state.get("cleaned_text") or "").strip()
    if not text:
        return {
            "error": "ml_classify_node requires non-empty cleaned_text in state or cleaned_text= kwarg."
        }

    try:
        label, confidence, p_fake, p_real = classify_cleaned_text(text, pipeline=pipeline)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive for graph robustness
        return {"error": f"ML classify failed: {exc}"}

    return {
        "ml_label": label,
        "ml_confidence": confidence,
        "ml_p_fake": p_fake,
        "ml_p_real": p_real,
    }


def describe_ml_classify_step() -> str:
    """
    Human-readable description of this node for docs or UI tooltips.

    Returns:
        Short summary string.
    """
    return "ML classify: TF-IDF + classifier verdict and scores."
