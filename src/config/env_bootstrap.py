"""
Runtime environment bootstrap for local runs and Streamlit Community Cloud.

- Loads ``.env`` from the repository root when ``python-dotenv`` is available
  (does not override variables already set in the process environment).
- Mirrors selected keys from ``st.secrets`` into ``os.environ`` when the app
  runs under Streamlit, so existing ``os.environ.get(...)`` call sites keep working.

Safe to call multiple times (e.g. on Streamlit reruns). Importing this module
does not import Streamlit until ``bootstrap_runtime_env()`` runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _repo_root() -> Path:
    return _REPO_ROOT


def merge_dotenv_over_empty_env_keys(env_path: Optional[Path] = None) -> None:
    """
    Copy values from a ``.env`` file into ``os.environ`` only for keys that are
    missing or whitespace-only.

    ``load_dotenv(..., override=False)`` skips any key already present in the
    environment, including ``GROQ_API_KEY=""`` from the shell or tooling. This
    fills those blanks from the file so a valid ``.env`` still applies.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    path = env_path if env_path is not None else (_repo_root() / ".env")
    if not path.is_file():
        return
    data = dotenv_values(path)
    for key in (
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "LLM_PROVIDER",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "HF_HOME",
        "HUGGINGFACE_HUB_TOKEN",
        "HF_TOKEN",
    ):
        if key not in data:
            continue
        raw = data[key]
        if raw is None:
            continue
        val = str(raw).strip()
        if not val:
            continue
        if (os.environ.get(key) or "").strip():
            continue
        os.environ[key] = val


def _load_dotenv_from_repo() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = _repo_root() / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    merge_dotenv_over_empty_env_keys(env_path)


def _env_nonempty(key: str) -> bool:
    return bool((os.environ.get(key) or "").strip())


def _set_env_if_unset(key: str, value: Optional[object]) -> None:
    if value is None:
        return
    s = str(value).strip()
    if not s:
        return
    if _env_nonempty(key):
        return
    os.environ[key] = s


def _inject_streamlit_secrets() -> None:
    try:
        import streamlit as st
    except ImportError:
        return
    secrets = getattr(st, "secrets", None)
    if secrets is None:
        return

    for key in (
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "LLM_PROVIDER",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "HF_HOME",
        "HUGGINGFACE_HUB_TOKEN",
        "HF_TOKEN",
    ):
        try:
            if key in secrets:
                _set_env_if_unset(key, secrets[key])
        except Exception:
            continue

    try:
        groq_block = secrets.get("groq")
        if isinstance(groq_block, dict):
            if not _env_nonempty("GROQ_API_KEY"):
                _set_env_if_unset(
                    "GROQ_API_KEY",
                    groq_block.get("api_key") or groq_block.get("API_KEY"),
                )
            if not _env_nonempty("GROQ_MODEL"):
                _set_env_if_unset(
                    "GROQ_MODEL",
                    groq_block.get("model") or groq_block.get("MODEL"),
                )
    except Exception:
        pass

    try:
        hf_block = secrets.get("huggingface") or secrets.get("hf")
        if isinstance(hf_block, dict) and not _env_nonempty("HUGGINGFACE_HUB_TOKEN"):
            _set_env_if_unset(
                "HUGGINGFACE_HUB_TOKEN",
                hf_block.get("token") or hf_block.get("HUB_TOKEN"),
            )
    except Exception:
        pass

    try:
        gem_block = secrets.get("gemini")
        if isinstance(gem_block, dict):
            if not _env_nonempty("GEMINI_API_KEY"):
                _set_env_if_unset(
                    "GEMINI_API_KEY",
                    gem_block.get("api_key") or gem_block.get("API_KEY"),
                )
            if not _env_nonempty("GEMINI_MODEL"):
                _set_env_if_unset(
                    "GEMINI_MODEL",
                    gem_block.get("model") or gem_block.get("MODEL"),
                )
    except Exception:
        pass


def bootstrap_runtime_env() -> None:
    """
    Load local ``.env`` and apply Streamlit Cloud secrets to the environment.

    Call from the Streamlit entrypoint before any code that reads ``GROQ_*``,
    ``GEMINI_*``, ``LLM_PROVIDER``, or Hugging Face tokens from ``os.environ``.
    """
    _load_dotenv_from_repo()
    _inject_streamlit_secrets()
    # Hugging Face libraries read HUGGINGFACE_HUB_TOKEN; some docs use HF_TOKEN.
    if _env_nonempty("HF_TOKEN") and not _env_nonempty("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"].strip()
