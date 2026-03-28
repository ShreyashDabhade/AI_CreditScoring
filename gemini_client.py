"""Shared Gemini client initialization for the credit scoring chatbot."""
from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL_CANDIDATES = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro-latest",
    "gemini-pro",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-flash-latest",
]

try:
    import google.generativeai as genai  # type: ignore
except Exception as exc:
    logger.critical("[INIT] Failed to import google.generativeai: %s", exc)
    raise

_model: Any | None = None
_model_name: str | None = None


def _find_working_model(api_key: str) -> Any:
    """Try available Gemini models and return the first one that works."""
    preferred_model = os.getenv("GEMINI_MODEL")
    genai.configure(api_key=api_key)

    try:
        available = [
            model.name
            for model in genai.list_models()
            if "generateContent" in getattr(model, "supported_generation_methods", [])
        ]
        logger.info("[GEMINI] Available models: %s", available)

        if preferred_model and preferred_model in available:
            logger.info("[GEMINI] Selected configured model: %s", preferred_model)
            return genai.GenerativeModel(preferred_model)

        for candidate in available:
            lowered = candidate.lower()
            if "flash" in lowered and "2.5" not in lowered:
                logger.info("[GEMINI] Selected model: %s", candidate)
                return genai.GenerativeModel(candidate)

        if available:
            logger.info("[GEMINI] Falling back to first available: %s", available[0])
            return genai.GenerativeModel(available[0])
    except Exception as exc:
        logger.warning("[GEMINI] Could not list models: %s. Trying candidates directly.", exc)

    candidate_names = [preferred_model] if preferred_model else []
    candidate_names.extend(_MODEL_CANDIDATES)
    seen: set[str] = set()
    for candidate in candidate_names:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            model = genai.GenerativeModel(candidate)
            model.generate_content("hi")
            logger.info("[GEMINI] Working model found: %s", candidate)
            return model
        except Exception as exc:
            logger.debug("[GEMINI] %s failed: %s", candidate, exc)

    raise RuntimeError(
        "No working Gemini model found. Check your API key and quota at "
        "https://aistudio.google.com/apikey"
    )


def get_model() -> Any:
    """Return the singleton Gemini model instance for the application."""
    global _model, _model_name

    if _model is not None:
        return _model

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY missing from .env file")

    try:
        _model = _find_working_model(api_key)
        _model_name = getattr(_model, "model_name", None) or "unavailable"
        logger.info("[GEMINI] Model ready: %s", _model_name)
        return _model
    except Exception as exc:
        logger.critical("[INIT] Gemini initialization FAILED: %s", exc)
        raise


def get_model_name() -> str:
    """Return the name of the currently active Gemini model."""
    if _model_name:
        return _model_name
    try:
        return getattr(get_model(), "model_name", "unavailable")
    except Exception:
        return "unavailable"
