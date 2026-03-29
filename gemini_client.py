"""Shared Gemini client helpers for the chatbot pipeline."""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_CANDIDATES = [
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro-latest",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash",
]

_model: Any | None = None
_model_name: str | None = None
_attempted = False


def _load_genai_module() -> Any:
    import google.generativeai as genai  # type: ignore

    return genai


def _find_working_model(api_key: str) -> Any:
    preferred_model = os.getenv("GEMINI_MODEL", "").strip()
    genai = _load_genai_module()
    genai.configure(api_key=api_key)

    candidate_names: list[str] = []
    if preferred_model:
        candidate_names.append(preferred_model)

    try:
        available = [
            model.name
            for model in genai.list_models()
            if "generateContent" in getattr(model, "supported_generation_methods", [])
        ]
        for candidate in available:
            if candidate not in candidate_names:
                candidate_names.append(candidate)
    except Exception as exc:
        logger.debug("Could not list Gemini models: %s", exc)

    for candidate in _MODEL_CANDIDATES:
        if candidate not in candidate_names:
            candidate_names.append(candidate)

    for candidate in candidate_names:
        if not candidate:
            continue
        try:
            model = genai.GenerativeModel(candidate)
            logger.info("Gemini model selected: %s", candidate)
            return model
        except Exception as exc:
            logger.debug("Gemini model candidate %s failed: %s", candidate, exc)

    raise RuntimeError("No working Gemini model could be initialized")


def get_model() -> Any:
    global _attempted, _model, _model_name

    if _model is not None:
        return _model
    if _attempted:
        raise RuntimeError("Gemini model unavailable")

    _attempted = True
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not configured")

    _model = _find_working_model(api_key)
    _model_name = getattr(_model, "model_name", None) or os.getenv("GEMINI_MODEL") or "unavailable"
    return _model


def get_model_name() -> str:
    if _model_name:
        return _model_name
    try:
        model = get_model()
        return getattr(model, "model_name", None) or "unavailable"
    except Exception:
        return "unavailable"


def is_available() -> bool:
    return get_model_name() != "unavailable"
