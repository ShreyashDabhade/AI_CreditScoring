from __future__ import annotations

import os
from pathlib import Path

from src.api.app import create_app


PROJECT_ROOT = Path(__file__).resolve().parent
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

_REAL_RUNTIME_REQUIRED_PATHS = (
    PROJECT_ROOT / "artifacts" / "full_feature_builder.joblib",
    PROJECT_ROOT / "artifacts" / "full_feature_builder.manifest.json",
    PROJECT_ROOT / "artifacts" / "full_model.joblib",
    PROJECT_ROOT / "artifacts" / "full_calibrator.joblib",
    PROJECT_ROOT / "artifacts" / "full_shap_explainer.joblib",
    PROJECT_ROOT / "artifacts" / "reduced_feature_builder.joblib",
    PROJECT_ROOT / "artifacts" / "reduced_feature_builder.manifest.json",
    PROJECT_ROOT / "artifacts" / "reduced_model.joblib",
    PROJECT_ROOT / "artifacts" / "reduced_calibrator.joblib",
    PROJECT_ROOT / "artifacts" / "reduced_shap_explainer.joblib",
    PROJECT_ROOT / "artifacts" / "model_fairness_audit_passed.joblib",
    PROJECT_ROOT / "data" / "processed" / "processed_artifact_manifest.json",
)


def _env_flag(name: str) -> bool | None:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return None
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return None


def _has_complete_real_runtime() -> bool:
    return all(path.exists() for path in _REAL_RUNTIME_REQUIRED_PATHS)


def _resolve_mock_mode() -> bool:
    explicit = _env_flag("MASTERMIND_MOCK_MODE")
    if explicit is not None:
        return explicit
    return not _has_complete_real_runtime()


app = create_app(mock_mode=_resolve_mock_mode())


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    app.run(host=host, port=port)
