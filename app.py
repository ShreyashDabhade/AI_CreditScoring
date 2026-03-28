from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from flask import Flask, abort, render_template, request, send_from_directory, url_for
from jinja2 import ChoiceLoader, FileSystemLoader
import pandas as pd

from src.api.app import RUNTIME_EXTENSION_KEY, _build_demo_config, create_app


PROJECT_ROOT = Path(__file__).resolve().parent
UI_STATIC_DIR = PROJECT_ROOT / "static"
UI_TEMPLATE_DIR = PROJECT_ROOT / "templates"
ANALYTICS_DIRS = {
    "eda": PROJECT_ROOT / "notebooks" / "eda_plots",
    "eval": PROJECT_ROOT / "notebooks" / "eval_plots",
    "shap": PROJECT_ROOT / "notebooks" / "shap_plots",
    "fairness": PROJECT_ROOT / "notebooks" / "fairness_plots",
}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_flag(name: str) -> bool | None:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return None
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return None


def _artifact_dir() -> Path:
    return Path(os.getenv("ARTIFACT_DIR", str(PROJECT_ROOT / "artifacts"))).resolve()


def _processed_dir() -> Path:
    return Path(
        os.getenv("DATA_PROCESSED_DIR", str(PROJECT_ROOT / "data" / "processed"))
    ).resolve()


def _real_runtime_required_paths() -> tuple[Path, ...]:
    artifact_dir = _artifact_dir()
    return (
        artifact_dir / "full_feature_builder.joblib",
        artifact_dir / "full_model.joblib",
        artifact_dir / "full_calibrator.joblib",
        artifact_dir / "full_shap_explainer.joblib",
        artifact_dir / "reduced_feature_builder.joblib",
        artifact_dir / "reduced_model.joblib",
        artifact_dir / "reduced_calibrator.joblib",
        artifact_dir / "reduced_shap_explainer.joblib",
        artifact_dir / "model_fairness_audit_passed.joblib",
    )


def _strict_runtime_sidecar_paths() -> tuple[Path, ...]:
    artifact_dir = _artifact_dir()
    processed_dir = _processed_dir()
    return (
        artifact_dir / "full_feature_builder.manifest.json",
        artifact_dir / "reduced_feature_builder.manifest.json",
        processed_dir / "processed_artifact_manifest.json",
    )


def _has_complete_real_runtime() -> bool:
    return all(path.exists() for path in _real_runtime_required_paths())


def _resolve_strict_artifacts() -> bool:
    explicit = _env_flag("MASTERMIND_STRICT_ARTIFACTS")
    if explicit is not None:
        return explicit
    return all(path.exists() for path in _strict_runtime_sidecar_paths())


def _resolve_mock_mode() -> bool:
    explicit = _env_flag("MASTERMIND_MOCK_MODE")
    if explicit is not None:
        return explicit
    return not _has_complete_real_runtime()


def _runtime(app: Flask) -> Any:
    return app.extensions[RUNTIME_EXTENSION_KEY]


def _build_health_snapshot(app: Flask) -> dict[str, Any]:
    runtime = _runtime(app)
    return {
        "status": "ok",
        "model_version": runtime.health_model_version,
        "fairness_audit_passed": runtime.model_fairness_audit_passed,
        "coverage_tiers_available": list(runtime.coverage_tiers_available),
    }


def _build_ui_config(app: Flask) -> dict[str, Any]:
    config = dict(_build_demo_config(_runtime(app)))
    routes = dict(config.get("routes", {}))
    routes.update(
        {
            "home": "/",
            "analyze": "/analyze",
            "status": "/status",
            "analytics": "/analytics",
        }
    )
    config["routes"] = routes
    return config


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def _load_optional_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _load_optional_csv(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    safe_df = df.where(pd.notna(df), None)
    return {
        "columns": list(df.columns),
        "rows": json.loads(safe_df.to_json(orient="records")),
    }


def _plot_entries(section: str) -> list[dict[str, str]]:
    directory = ANALYTICS_DIRS[section]
    if not directory.exists():
        return []
    return [
        {
            "filename": path.name,
            "title": path.stem.replace("_", " ").title(),
            "url": url_for("ui_artifact", section=section, filename=path.name),
        }
        for path in sorted(directory.glob("*.png"))
    ]


def _build_analytics_config() -> dict[str, Any]:
    fairness_dir = ANALYTICS_DIRS["fairness"]
    return {
        "plots": {section: _plot_entries(section) for section in ANALYTICS_DIRS},
        "qualityReport": _load_optional_json(PROJECT_ROOT / "data" / "data_quality_report.json"),
        "championReport": _load_optional_text(_artifact_dir() / "champion_report.txt"),
        "fairnessTables": {
            family: table
            for family in ("primary", "secondary", "tertiary")
            if (table := _load_optional_csv(fairness_dir / f"audit_{family}.csv")) is not None
        },
    }


def _render_ui_page(app: Flask, template_name: str, *, page_title: str, active_nav: str):
    return render_template(
        template_name,
        page_title=page_title,
        active_nav=active_nav,
        ui_config=_build_ui_config(app),
        health_snapshot=_build_health_snapshot(app),
    )


def _install_ui(app: Flask) -> None:
    ui_loader = FileSystemLoader(str(UI_TEMPLATE_DIR))
    existing_loader = app.jinja_loader
    app.jinja_loader = (
        ChoiceLoader([ui_loader, existing_loader])
        if existing_loader is not None
        else ui_loader
    )

    def home():
        return _render_ui_page(
            app,
            "index.html",
            page_title="MasterMind Credit Scoring",
            active_nav="home",
        )

    def analyze():
        return _render_ui_page(
            app,
            "analyze.html",
            page_title="Credit Analysis",
            active_nav="analyze",
        )

    def demo():
        if request.path == "/demo":
            return analyze()
        return home()

    def status_page():
        return _render_ui_page(
            app,
            "status.html",
            page_title="System Status",
            active_nav="status",
        )

    def analytics():
        return render_template(
            "analytics.html",
            page_title="Analytics Dashboard",
            active_nav="analytics",
            ui_config=_build_ui_config(app),
            health_snapshot=_build_health_snapshot(app),
            analytics_config=_build_analytics_config(),
        )

    def ui_static(filename: str):
        return send_from_directory(str(UI_STATIC_DIR), filename)

    def ui_artifact(section: str, filename: str):
        if section not in ANALYTICS_DIRS:
            abort(404)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", filename):
            abort(403)
        if not filename.lower().endswith(".png"):
            abort(403)
        return send_from_directory(str(ANALYTICS_DIRS[section]), filename)

    app.view_functions["demo"] = demo
    app.add_url_rule("/", endpoint="home", view_func=home)
    app.add_url_rule("/analyze", endpoint="analyze", view_func=analyze)
    app.add_url_rule("/status", endpoint="status_page", view_func=status_page)
    app.add_url_rule("/analytics", endpoint="analytics", view_func=analytics)
    app.add_url_rule("/ui-static/<path:filename>", endpoint="ui_static", view_func=ui_static)
    app.add_url_rule(
        "/ui-artifacts/<section>/<filename>",
        endpoint="ui_artifact",
        view_func=ui_artifact,
    )


app = create_app(
    artifact_dir=str(_artifact_dir()),
    processed_dir=str(_processed_dir()),
    mock_mode=_resolve_mock_mode(),
    strict_artifacts=_resolve_strict_artifacts(),
)
_install_ui(app)


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    app.run(host=host, port=port)
