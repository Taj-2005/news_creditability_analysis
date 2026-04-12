"""
Groq API client for agent reasoning (query planning, verification, reporting).

Configuration via environment variables:

- ``GROQ_API_KEY`` (required for live calls): API key from https://console.groq.com/
- ``GROQ_MODEL`` (optional): chat model id, default ``llama-3.1-8b-instant``

If ``python-dotenv`` is installed, ``.env`` in the working directory is loaded on first use.
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

        load_dotenv()
    except ImportError:
        pass
    _DOTENV_LOADED = True


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def generate(prompt: str) -> str:
    """
    Run a single-turn chat completion against the Groq API.

    Args:
        prompt: Full user message (instructions + content).

    Returns:
        Trimmed assistant text.

    Raises:
        RuntimeError: If ``GROQ_API_KEY`` is missing or the API returns no content.
        Exception: Propagates client/network errors from ``groq``.
    """
    _maybe_load_dotenv()
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file "
            "(see README) and restart the process."
        )

    model = (os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL).strip()
    from groq import Groq

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt.strip()}],
        max_tokens=2048,
        temperature=0.2,
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
