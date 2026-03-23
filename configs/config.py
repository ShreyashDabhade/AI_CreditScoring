"""Compatibility loader for the shared project configuration module."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_ARTIFACT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "configs" / "config.py"
)

if not _ARTIFACT_CONFIG_PATH.exists():
    raise ModuleNotFoundError(
        f"Could not locate shared configuration at {_ARTIFACT_CONFIG_PATH}"
    )

_spec = spec_from_file_location("_artifact_shared_config", _ARTIFACT_CONFIG_PATH)
if _spec is None or _spec.loader is None:
    raise ModuleNotFoundError(
        f"Could not load shared configuration from {_ARTIFACT_CONFIG_PATH}"
    )

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

__all__ = [name for name in dir(_module) if not name.startswith("_")]

for _name in __all__:
    globals()[_name] = getattr(_module, _name)
