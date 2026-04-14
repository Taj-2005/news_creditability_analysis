"""
LLM client for agent reasoning (query planning, verification, reporting).

Primary backend is **Groq**. Optional backend is **Gemini** via ``google-generativeai``.

Configuration via environment variables:

- ``GROQ_API_KEY`` (required for live calls): API key from https://console.groq.com/
- ``GROQ_MODEL`` (optional): chat model id, default ``llama-3.1-8b-instant``
- ``LLM_PROVIDER`` (optional): ``auto`` (default), ``groq``, or ``gemini``
- ``GEMINI_API_KEY`` / ``GEMINI_MODEL`` (optional): Gemini configuration

If Gemini is selected and errors, ``generate()`` falls back to Groq when ``GROQ_API_KEY`` is set.

If ``python-dotenv`` is installed, ``.env`` at the repository root is loaded on first use,
including filling keys that exist in the environment but are blank-only.
"""

from __future__ import annotations

import os

_DOTENV_LOADED = False


def _maybe_load_dotenv() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        from src.config.env_bootstrap import merge_dotenv_over_empty_env_keys

        root = Path(__file__).resolve().parent.parent.parent
        env_file = root / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)
        merge_dotenv_over_empty_env_keys(env_file)
    except ImportError:
        pass
    _DOTENV_LOADED = True


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"


def _generate_gemini(prompt: str, *, temperature: float, max_tokens: int) -> str:
    """
    Optional Gemini backend.

    Requires:
      - ``GEMINI_API_KEY``
      - package ``google-generativeai`` installed
    """
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    model = (os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Gemini backend requested but google-generativeai is not installed."
        ) from exc
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model)
    resp = m.generate_content(
        prompt.strip(),
        generation_config={
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
        },
    )
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty text.")
    return text


def generate(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Run a single-turn chat completion against the Groq API.

    Args:
        prompt: Full user message (instructions + content).
        temperature: Sampling temperature (use ``0.0`` for deterministic verification).
        max_tokens: Maximum completion tokens.

    Returns:
        Trimmed assistant text.

    Raises:
        RuntimeError: If ``GROQ_API_KEY`` is missing or the API returns no content.
        Exception: Propagates client/network errors from ``groq``.
    """
    _maybe_load_dotenv()
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    # Provider selection: Groq by default; Gemini optional when configured.
    # If Gemini is selected but fails, fall back to Groq when available.
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    gem_key = (os.environ.get("GEMINI_API_KEY") or "").strip()

    prefer_gemini = provider == "gemini"
    auto_gemini = (provider in ("", "auto")) and (not groq_key) and bool(gem_key)

    if prefer_gemini or auto_gemini:
        try:
            return _generate_gemini(prompt, temperature=temperature, max_tokens=max_tokens)
        except Exception:
            if not groq_key:
                raise

    api_key = groq_key
    if not api_key:
        raise RuntimeError(
            "No LLM API key configured. Set GROQ_API_KEY (default) or set LLM_PROVIDER=gemini with GEMINI_API_KEY. "
            "Use `.env` at the repository root or Streamlit Cloud Secrets."
        )

    model = (os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL).strip()
    from groq import Groq

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt.strip()}],
        max_tokens=max_tokens,
        temperature=float(temperature),
    )
    choice = completion.choices[0].message
    text = (choice.content or "").strip()
    if not text:
        raise RuntimeError("Groq returned an empty assistant message.")
    return text


def is_configured() -> bool:
    """Return True if ``GROQ_API_KEY`` appears to be set (after optional dotenv load)."""
    _maybe_load_dotenv()
    return bool((os.environ.get("GROQ_API_KEY") or "").strip())
