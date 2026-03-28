"""Compatibility shim for running src modules as standalone scripts."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT_MODULE = Path(__file__).resolve().parents[1] / "gemini_client.py"
_SPEC = importlib.util.spec_from_file_location("_root_gemini_client", _ROOT_MODULE)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - defensive guard
    raise ImportError(f"Unable to load root gemini_client module from {_ROOT_MODULE}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

get_model = _MODULE.get_model
get_model_name = _MODULE.get_model_name

