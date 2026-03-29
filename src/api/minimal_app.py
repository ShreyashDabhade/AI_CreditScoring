"""Minimal Flask wrapper for the saved scoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

from configs.config import (
    APPROVE_THRESHOLD,
    ARTIFACT_DIR,
    DATA_DIR,
    DECLINE_THRESHOLD,
    DRIFT_ALERT_THRESHOLD,
    DRIFT_BASELINE_PD_MEAN,
    DRIFT_LOOKBACK_WINDOW,
    DRIFT_MIN_SAMPLE_SIZE,
    DRIFT_WATCH_THRESHOLD,
    FAIRNESS_AUDIT_VERSION,
    MODEL_VERSIONS,
)
from src.api.app import _build_demo_config, _build_demo_seed_payload
from src.chatbot import chat as chatbot_chat, is_gemini_available
from src.data_pipeline import TRAIN_SCHEMA, enforce_schema
from src.db_manager import (
    ALLOWED_STATUS_VALUES,
    DEFAULT_DB_FILENAME,
    create_application,
    get_application_by_id,
    init_database,
    list_application_change_history,
    list_applications,
    list_score_history_for_application,
    resolve_database_path,
    save_application_change_log,
    save_score_run,
    update_application,
)
from src.explainability import render_reason
from src.feature_engineering import (
    APPLICATION_REQUIRED_INPUT_COLS,
    BUREAU_AGG_COLS,
    CREDIT_CARD_AGG_COLS,
    CATEGORICAL_MODEL_COLS,
    INSTALLMENTS_AGG_COLS,
    POS_CASH_AGG_COLS,
    PREVIOUS_AGG_COLS,
    build_full,
    build_reduced,
)
from src.model_monitoring import compute_probability_drift_snapshot

RUNTIME_EXTENSION_KEY = "minimal_scoring_runtime"
APPLICATION_DB_PATH_KEY = "minimal_scoring_application_db_path"
SUPPORTED_TIERS = ("FULL", "REDUCED")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_TEMPLATE_DIR = PROJECT_ROOT / "templates"
UI_STATIC_DIR = PROJECT_ROOT / "static"
SECTION_FIELD_MAP: Mapping[str, tuple[str, ...]] = {
    "bureau_agg": tuple(BUREAU_AGG_COLS),
    "previous_agg": tuple(PREVIOUS_AGG_COLS),
    "installments_agg": tuple(INSTALLMENTS_AGG_COLS),
    "pos_cash_agg": tuple(POS_CASH_AGG_COLS),
    "credit_card_agg": tuple(CREDIT_CARD_AGG_COLS),
}
NUMERIC_APPLICATION_FIELDS = tuple(
    field for field in APPLICATION_REQUIRED_INPUT_COLS if field not in set(CATEGORICAL_MODEL_COLS)
)
NUMERIC_REQUEST_FIELDS = (
    ("AMT_INCOME_TOTAL",)
    + NUMERIC_APPLICATION_FIELDS
    + tuple(field for fields in SECTION_FIELD_MAP.values() for field in fields)
)
SIMULATOR_NEGLIGIBLE_DELTA = 0.0025
WHAT_IF_FIELD_SPECS = (
    {
        "section": "application",
        "name": "AMT_CREDIT",
        "label": "Credit Amount",
        "step": "1000",
        "min": "50000",
        "max": "1500000",
        "control": "slider",
        "value_kind": "amount",
        "decimals": 0,
    },
    {
        "section": "application",
        "name": "AMT_ANNUITY",
        "label": "Annuity",
        "step": "500",
        "min": "5000",
        "max": "100000",
        "control": "slider",
        "value_kind": "amount",
        "decimals": 0,
    },
    {
        "section": "application",
        "name": "AMT_INCOME_TOTAL_CAPPED",
        "label": "Income Total",
        "step": "5000",
        "min": "25000",
        "max": "500000",
        "control": "slider",
        "value_kind": "amount",
        "decimals": 0,
    },
    {
        "section": "application",
        "name": "EXT_SOURCE_1",
        "label": "External Source 1",
        "step": "0.01",
        "min": "0",
        "max": "1",
        "control": "slider",
        "value_kind": "ratio",
        "decimals": 2,
    },
    {
        "section": "application",
        "name": "EXT_SOURCE_2",
        "label": "External Source 2",
        "step": "0.01",
        "min": "0",
        "max": "1",
        "control": "slider",
        "value_kind": "ratio",
        "decimals": 2,
    },
    {
        "section": "application",
        "name": "EXT_SOURCE_3",
        "label": "External Source 3",
        "step": "0.01",
        "min": "0",
        "max": "1",
        "control": "slider",
        "value_kind": "ratio",
        "decimals": 2,
    },
    {
        "section": "bureau_agg",
        "name": "BUREAU_DEBT_TO_CREDIT_RATIO",
        "label": "Bureau Debt / Credit Ratio",
        "step": "0.01",
        "min": "0",
        "control": "number",
    },
    {
        "section": "previous_agg",
        "name": "PREV_APPROVAL_RATE",
        "label": "Previous Approval Rate",
        "step": "0.01",
        "min": "0",
        "max": "1",
        "control": "number",
    },
    {
        "section": "installments_agg",
        "name": "INST_DPD_MEAN",
        "label": "Installment DPD Mean",
        "step": "1",
        "min": "0",
        "control": "number",
    },
    {
        "section": "pos_cash_agg",
        "name": "POS_DPD_MEAN",
        "label": "POS Cash DPD Mean",
        "step": "0.5",
        "min": "0",
        "control": "number",
    },
    {
        "section": "credit_card_agg",
        "name": "CC_UTILIZATION_MEAN",
        "label": "Credit Card Utilization Mean",
        "step": "0.01",
        "min": "0",
        "control": "number",
    },
)


@dataclass(frozen=True)
class MinimalRuntime:
    artifact_dir: str
    processed_dir: str
    application_db_path: str
    processed_manifest: Mapping[str, Any]
    income_cap: float
    full_builder: Any
    full_model: Any
    full_calibrator: Any
    full_shap_explainer: Any
    reduced_builder: Any
    reduced_model: Any
    reduced_calibrator: Any
    reduced_shap_explainer: Any
    model_fairness_audit_passed: bool
    health_model_version: str
    tier_model_versions: Mapping[str, str]
    coverage_tiers_available: tuple[str, ...]
    reproducibility_report: Mapping[str, Any]
    mock_mode: bool


def create_app(
    artifact_dir: str | None = None,
    processed_dir: str | None = None,
    application_db_path: str | None = None,
) -> Flask:
    resolved_artifact_dir = os.path.abspath(artifact_dir or ARTIFACT_DIR)
    resolved_processed_dir = os.path.abspath(processed_dir or DATA_DIR)
    resolved_db_path = resolve_database_path(
        application_db_path
        or os.getenv("MASTERMIND_DB_PATH")
        or os.path.join(os.path.dirname(resolved_processed_dir), DEFAULT_DB_FILENAME)
    )
    init_database(resolved_db_path)

    runtime = _load_runtime(
        artifact_dir=resolved_artifact_dir,
        processed_dir=resolved_processed_dir,
        application_db_path=resolved_db_path,
    )

    app = Flask(
        __name__,
        template_folder=str(UI_TEMPLATE_DIR),
        static_folder=str(UI_STATIC_DIR),
    )
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.extensions[RUNTIME_EXTENSION_KEY] = runtime
    app.extensions[APPLICATION_DB_PATH_KEY] = resolved_db_path
    app.config["APPLICATION_DB_PATH"] = resolved_db_path

    @app.get("/health")
    def health():
        loaded_runtime = _get_runtime(app)
        return jsonify(
            {
                "status": "ok",
                "model_version": loaded_runtime.health_model_version,
                "coverage_tiers_available": list(SUPPORTED_TIERS),
                "fairness_audit_passed": loaded_runtime.model_fairness_audit_passed,
            }
        )

    @app.post("/score")
    def score():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error_code": "bad_request", "message": "Request body must be a JSON object."}), 400

        try:
            response_body = _score_payload(_get_runtime(app), payload)
        except ValueError as exc:
            return jsonify({"error_code": "bad_request", "message": str(exc)}), 400
        except Exception:
            app.logger.exception("Minimal scoring request failed.")
            return jsonify({"error_code": "internal_error", "message": "Scoring failed."}), 500

        return jsonify(response_body)

    @app.get("/api/chat/health")
    def api_chat_health():
        return jsonify({"status": "ok", "gemini_available": is_gemini_available()})

    @app.post("/api/chat")
    def api_chat():
        try:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return jsonify({"error": "bad_request", "message": "JSON payload required"}), 400

            message = str(payload.get("message", "")).strip()
            if not message:
                return jsonify({"error": "bad_request", "message": "message is required"}), 400

            context = payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}
            result = chatbot_chat(
                message=message,
                score_data=context.get("score_data"),
                shap_values=context.get("shap_values"),
                page_context=context.get("page_context"),
                report_context=context.get("report_context"),
                conversation_history=payload.get("history", []) if isinstance(payload.get("history"), list) else [],
            )
            return jsonify(result)
        except Exception as exc:
            app.logger.exception("Minimal chat request failed.")
            return jsonify({"error": "internal_error", "message": str(exc)}), 500

    @app.post("/analyst/applications/<int:application_id>/simulate")
    def analyst_application_simulate(application_id: int):
        payload = request.get_json(silent=True)
        requested_changes = payload.get("changes", payload) if isinstance(payload, dict) else {}
        if not isinstance(requested_changes, dict):
            return jsonify({"error": "bad_request", "message": "Simulation changes must be an object."}), 400

        try:
            result = _simulate_saved_application(_get_runtime(app), application_id, requested_changes)
        except LookupError:
            return jsonify({"error": "not_found", "message": "Application not found."}), 404
        except ValueError as exc:
            return jsonify({"error": "validation_error", "message": str(exc)}), 400
        except Exception as exc:
            app.logger.exception("Minimal simulation request failed for id=%s", application_id)
            return jsonify({"error": "simulation_failed", "message": str(exc)}), 500
        return jsonify(result)

    @app.get("/")
    def home():
        return render_template(
            "index.html",
            page_title="MasterMind Credit Scoring",
            active_nav="home",
            ui_config=_build_ui_config(runtime),
            health_snapshot=_build_health_snapshot(runtime),
        )

    @app.get("/analyze")
    def analyze():
        return render_template(
            "analyze.html",
            page_title="Credit Analysis",
            active_nav="analyze",
            ui_config=_build_ui_config(runtime),
            health_snapshot=_build_health_snapshot(runtime),
        )

    @app.get("/status")
    def status_page():
        return render_template(
            "status.html",
            page_title="System Status",
            active_nav="status",
            ui_config=_build_ui_config(runtime),
            health_snapshot=_build_health_snapshot(runtime),
            drift_snapshot=_build_drift_snapshot(runtime.application_db_path),
        )

    @app.get("/analytics")
    def analytics():
        return render_template(
            "analytics.html",
            page_title="Analytics Dashboard",
            active_nav="analytics",
            ui_config=_build_ui_config(runtime),
            health_snapshot=_build_health_snapshot(runtime),
            analytics_config=_build_analytics_config(runtime),
        )

    @app.get("/applications/new")
    def application_new():
        return render_template(
            "applications_new.html",
            page_title="New Application",
            active_nav="applications",
            ui_config=_build_ui_config(runtime),
            health_snapshot=_build_health_snapshot(runtime),
            allowed_statuses=sorted(ALLOWED_STATUS_VALUES),
            error_message=request.args.get("error"),
        )

    @app.post("/applications")
    def application_create():
        try:
            created = create_application(
                runtime.application_db_path,
                **_build_create_application_args(runtime),
            )
        except ValueError as exc:
            return render_template(
                "applications_new.html",
                page_title="New Application",
                active_nav="applications",
                ui_config=_build_ui_config(runtime),
                health_snapshot=_build_health_snapshot(runtime),
                allowed_statuses=sorted(ALLOWED_STATUS_VALUES),
                error_message=str(exc),
            ), 400
        return redirect(url_for("analyst_application_detail", application_id=created["id"]))

    @app.get("/analyst")
    def analyst_dashboard():
        applications = [_application_view_model(item) for item in list_applications(runtime.application_db_path)]
        selected_id = request.args.get("selected", type=int)
        selected_application = None
        if applications:
            if selected_id is not None:
                selected_application = next((item for item in applications if item["id"] == selected_id), None)
            if selected_application is None:
                selected_application = applications[0]
        return render_template(
            "analyst_dashboard.html",
            page_title="Analyst Command Center",
            active_nav="analyst",
            ui_config=_build_ui_config(runtime),
            health_snapshot=_build_health_snapshot(runtime),
            applications=applications,
            status_counts=_status_counts(applications),
            dashboard_stats=_dashboard_stats(applications),
            selected_application=selected_application,
        )

    @app.get("/analyst/applications")
    def analyst_applications():
        applications = [_application_view_model(item) for item in list_applications(runtime.application_db_path)]
        if request.args.get("format", "").strip().lower() == "json":
            return jsonify({"applications": applications, "health_snapshot": _build_health_snapshot(runtime)})
        return render_template(
            "analyst_applications.html",
            page_title="Analyst Applications",
            active_nav="analyst",
            ui_config=_build_ui_config(runtime),
            health_snapshot=_build_health_snapshot(runtime),
            applications=applications,
        )

    @app.get("/analyst/applications/<int:application_id>")
    def analyst_application_detail(application_id: int):
        application = _application_view_model(_load_application_or_404(runtime, application_id))
        return render_template(
            "analyst_application_detail.html",
            page_title=f"Edit Application #{application_id}",
            active_nav="analyst",
            ui_config=_build_ui_config(runtime),
            health_snapshot=_build_health_snapshot(runtime),
            allowed_statuses=sorted(ALLOWED_STATUS_VALUES),
            application=application,
            score_history=[
                _score_run_view_model(item)
                for item in list_score_history_for_application(runtime.application_db_path, application_id)
            ],
            change_history=[
                _change_log_view_model(item)
                for item in list_application_change_history(runtime.application_db_path, application_id, limit=8)
            ],
        )

    @app.post("/analyst/applications/<int:application_id>/update")
    def analyst_application_update(application_id: int):
        original_application = _application_view_model(_load_application_or_404(runtime, application_id))
        try:
            updated = update_application(
                runtime.application_db_path,
                application_id,
                **_build_update_application_args(runtime),
            )
        except ValueError as exc:
            return redirect(url_for("analyst_application_detail", application_id=application_id, error=str(exc)))

        if updated is None:
            abort(404)
        updated_view = _application_view_model(updated)
        if change_summary := _build_change_summary(original_application, updated_view):
            save_application_change_log(
                runtime.application_db_path,
                application_id=application_id,
                change_summary_json=change_summary,
            )
        return redirect(url_for("analyst_application_detail", application_id=application_id))

    @app.post("/analyst/applications/<int:application_id>/analyze")
    def analyst_application_analyze(application_id: int):
        try:
            _analyze_saved_application(runtime, application_id)
        except ValueError as exc:
            return redirect(url_for("analyst_application_detail", application_id=application_id, error=str(exc)))
        except Exception:
            app.logger.exception("Persisted application analysis failed for id=%s", application_id)
            return redirect(
                url_for(
                    "analyst_application_detail",
                    application_id=application_id,
                    error="Analysis failed. Review the saved payload and runtime artifacts, then try again.",
                )
            )
        return redirect(url_for("analyst_application_report", application_id=application_id))

    @app.get("/analyst/applications/<int:application_id>/report")
    def analyst_application_report(application_id: int):
        application = _application_view_model(_load_application_or_404(runtime, application_id))
        score_history = [
            _score_run_view_model(item)
            for item in list_score_history_for_application(runtime.application_db_path, application_id)
        ]
        latest_score_run = score_history[0] if score_history else None
        health_snapshot = _build_health_snapshot(runtime)
        report = _build_report_context(application, latest_score_run, score_history, health_snapshot)
        return render_template(
            "analyst_application_report.html",
            page_title=f"Application Report #{application_id}",
            active_nav="analyst",
            ui_config=_build_ui_config(runtime),
            health_snapshot=health_snapshot,
            application=application,
            latest_score_run=latest_score_run,
            score_history=score_history,
            report=report,
            report_chat_context=_build_report_chat_context(application, report),
        )

    @app.get("/ui-static/<path:filename>")
    def ui_static(filename: str):
        return send_from_directory(str(UI_STATIC_DIR), filename)

    return app


def _get_runtime(app: Flask) -> MinimalRuntime:
    runtime = app.extensions.get(RUNTIME_EXTENSION_KEY)
    if not isinstance(runtime, MinimalRuntime):
        raise RuntimeError("Minimal scoring runtime is not initialized.")
    return runtime


def _load_runtime(*, artifact_dir: str, processed_dir: str, application_db_path: str) -> MinimalRuntime:
    fairness_path = os.path.join(artifact_dir, "model_fairness_audit_passed.joblib")
    fairness_value = False
    if os.path.exists(fairness_path):
        fairness_value = bool(joblib.load(fairness_path))

    income_cap = float(joblib.load(os.path.join(processed_dir, "income_cap.joblib")))
    processed_manifest = _load_optional_json(os.path.join(processed_dir, "processed_artifact_manifest.json"))
    reproducibility_report = _load_optional_json(os.path.join(artifact_dir, "reproducibility_report.json"))
    health_model_version = _resolve_model_version(
        reproducibility_report,
        preferred_keys=("deployed_model_version", "health_model_version", "model_version", "champion_model_version", "full_model_version"),
        fallback=MODEL_VERSIONS["full"],
    )
    tier_model_versions = {
        "FULL": _resolve_model_version(
            reproducibility_report,
            preferred_keys=("full_model_version", "full_deployed_model_version", "full_version"),
            fallback=MODEL_VERSIONS["full"],
        ),
        "REDUCED": _resolve_model_version(
            reproducibility_report,
            preferred_keys=("reduced_model_version", "reduced_deployed_model_version", "reduced_version"),
            fallback=MODEL_VERSIONS["reduced"],
        ),
    }
    return MinimalRuntime(
        artifact_dir=artifact_dir,
        processed_dir=processed_dir,
        application_db_path=application_db_path,
        processed_manifest=processed_manifest,
        income_cap=income_cap,
        full_builder=joblib.load(os.path.join(artifact_dir, "full_feature_builder.joblib")),
        full_model=joblib.load(os.path.join(artifact_dir, "full_model.joblib")),
        full_calibrator=joblib.load(os.path.join(artifact_dir, "full_calibrator.joblib")),
        full_shap_explainer=_load_optional_joblib(artifact_dir, "full_shap_explainer.joblib"),
        reduced_builder=joblib.load(os.path.join(artifact_dir, "reduced_feature_builder.joblib")),
        reduced_model=joblib.load(os.path.join(artifact_dir, "reduced_model.joblib")),
        reduced_calibrator=joblib.load(os.path.join(artifact_dir, "reduced_calibrator.joblib")),
        reduced_shap_explainer=_load_optional_joblib(artifact_dir, "reduced_shap_explainer.joblib"),
        model_fairness_audit_passed=fairness_value,
        health_model_version=health_model_version,
        tier_model_versions=tier_model_versions,
        coverage_tiers_available=SUPPORTED_TIERS,
        reproducibility_report=reproducibility_report,
        mock_mode=False,
    )


def _resolve_tier(payload: Mapping[str, Any]) -> str:
    explicit = payload.get("tier")
    if explicit is not None:
        tier = str(explicit).upper()
        if tier not in SUPPORTED_TIERS:
            raise ValueError("tier must be FULL or REDUCED")
        return tier

    if any(section in payload for section in SECTION_FIELD_MAP):
        return "FULL"
    return "REDUCED"


def _build_input_frame(payload: Mapping[str, Any], runtime: MinimalRuntime, tier: str) -> pd.DataFrame:
    application = payload.get("application")
    if not isinstance(application, dict):
        raise ValueError("application must be an object")

    row = dict(application)
    _prepare_application_fields(row, runtime.income_cap)

    if tier == "FULL":
        for section_name, required_fields in SECTION_FIELD_MAP.items():
            section_payload = payload.get(section_name)
            if not isinstance(section_payload, dict):
                raise ValueError(f"{section_name} must be an object for FULL scoring")
            missing_fields = [field for field in required_fields if field not in section_payload]
            if missing_fields:
                raise ValueError(f"{section_name} is missing required fields: {sorted(missing_fields)}")
            row.update(section_payload)

    frame = pd.DataFrame([row])
    frame = enforce_schema(frame, TRAIN_SCHEMA)
    _coerce_numeric_columns(frame)
    _coerce_categorical_columns(frame)

    missing_application_fields = [
        field for field in APPLICATION_REQUIRED_INPUT_COLS if field not in frame.columns
    ]
    if missing_application_fields:
        raise ValueError(
            f"application is missing required fields: {sorted(missing_application_fields)}"
        )

    return frame


def _prepare_application_fields(row: dict[str, Any], income_cap: float) -> None:
    if "AMT_INCOME_TOTAL_CAPPED" not in row:
        if "AMT_INCOME_TOTAL" not in row:
            raise ValueError(
                "application must include AMT_INCOME_TOTAL_CAPPED or AMT_INCOME_TOTAL"
            )
        income = float(row["AMT_INCOME_TOTAL"])
        row["AMT_INCOME_TOTAL_CAPPED"] = min(income, income_cap)

    if "DAYS_EMPLOYED_ANOM" not in row:
        if "DAYS_EMPLOYED" not in row:
            raise ValueError("application must include DAYS_EMPLOYED_ANOM or DAYS_EMPLOYED")
        days_employed = float(row["DAYS_EMPLOYED"])
        row["DAYS_EMPLOYED_ANOM"] = int(days_employed == 365243)
        if days_employed == 365243:
            row["DAYS_EMPLOYED"] = np.nan


def _coerce_numeric_columns(frame: pd.DataFrame) -> None:
    for field_name in NUMERIC_REQUEST_FIELDS:
        if field_name not in frame.columns:
            continue
        frame[field_name] = pd.to_numeric(frame[field_name], errors="raise")


def _coerce_categorical_columns(frame: pd.DataFrame) -> None:
    for field_name in CATEGORICAL_MODEL_COLS:
        if field_name not in frame.columns:
            continue
        frame[field_name] = frame[field_name].astype("object")


def _score_probability(runtime: MinimalRuntime, input_df: pd.DataFrame, tier: str) -> float:
    if tier == "FULL":
        features = build_full(input_df, runtime.full_builder, raw_dir=None)
        return _predict_and_calibrate(runtime.full_model, runtime.full_calibrator, features)

    features = build_reduced(input_df, runtime.reduced_builder)
    return _predict_and_calibrate(runtime.reduced_model, runtime.reduced_calibrator, features)


def _predict_and_calibrate(model: Any, calibrator: Any, features: pd.DataFrame) -> float:
    probs = np.asarray(model.predict_proba(features), dtype=float)
    if probs.ndim != 2 or probs.shape[0] != len(features) or probs.shape[1] < 2:
        raise RuntimeError("predict_proba must return shape (n_rows, >=2)")
    raw_pd = probs[:, 1]
    calibrated = np.asarray(calibrator.predict(raw_pd), dtype=float).reshape(-1)
    if calibrated.size != len(features):
        raise RuntimeError("calibrator predict must return one value per row")
    return float(np.clip(calibrated[0], 0.0, 1.0))


def _decision_from_pd(probability_of_default: float) -> str:
    if probability_of_default < APPROVE_THRESHOLD:
        return "APPROVE"
    if probability_of_default < DECLINE_THRESHOLD:
        return "REVIEW"
    return "DECLINE"


def _score_payload(runtime: MinimalRuntime, payload: Mapping[str, Any]) -> dict[str, Any]:
    tier = _resolve_tier(payload)
    input_df = _build_input_frame(payload, runtime, tier)
    if tier == "FULL":
        features = build_full(input_df, runtime.full_builder, raw_dir=None)
        probability = _predict_and_calibrate(runtime.full_model, runtime.full_calibrator, features)
    else:
        features = build_reduced(input_df, runtime.reduced_builder)
        probability = _predict_and_calibrate(runtime.reduced_model, runtime.reduced_calibrator, features)

    decision = _decision_from_pd(probability)
    return {
        "coverage_tier": tier,
        "probability_of_default": probability,
        "decision": decision,
        "escalate": decision == "REVIEW",
        "top_5_explanations": _default_top_5_explanations(features.columns),
        "model_version": runtime.tier_model_versions[tier],
        "calibrated": True,
        "model_fairness_audit_passed": runtime.model_fairness_audit_passed,
        "fairness_audit_version": FAIRNESS_AUDIT_VERSION,
    }


def _default_top_5_explanations(feature_names: Any) -> list[dict[str, str]]:
    ordered_names = list(feature_names)[:5]
    if len(ordered_names) < 5:
        ordered_names.extend(f"fallback_feature_{index}" for index in range(len(ordered_names), 5))
    return [{"feature": name, "reason": render_reason(name)} for name in ordered_names[:5]]


def _load_optional_joblib(artifact_dir: str, filename: str) -> Any:
    path = os.path.join(artifact_dir, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _load_optional_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _resolve_model_version(
    report: Mapping[str, Any],
    *,
    preferred_keys: tuple[str, ...],
    fallback: str,
) -> str:
    for key in preferred_keys:
        value = report.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def _parse_application_payload_json(application: Mapping[str, Any]) -> dict[str, Any]:
    raw_payload = application.get("application_payload_json")
    if not isinstance(raw_payload, str):
        raise ValueError("stored application payload is missing")
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise ValueError("stored application payload must be a JSON object")
    return payload


def _validate_saved_payload(runtime: MinimalRuntime, payload: Mapping[str, Any]) -> str:
    tier = _resolve_tier(payload)
    _build_input_frame(payload, runtime, tier)
    return tier


def _coerce_simulator_numeric_value(raw_value: Any, field_name: str) -> float:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required for simulation")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _clone_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _build_simulation_changes(
    original_payload: Mapping[str, Any],
    requested_changes: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _clone_payload(original_payload)
    changes: list[dict[str, Any]] = []

    for field_name, raw_value in requested_changes.items():
        matched_section = None
        for section_name, section_fields in SECTION_FIELD_MAP.items():
            if field_name in section_fields:
                matched_section = section_name
                break
        if matched_section is None:
            matched_section = "application"

        section = payload.setdefault(matched_section, {})
        if not isinstance(section, dict):
            raise ValueError(f"{matched_section} cannot be simulated for this payload")

        next_value = _coerce_simulator_numeric_value(raw_value, field_name)
        before_value = section.get(field_name)
        section[field_name] = next_value
        if before_value != next_value:
            changes.append(
                {
                    "label": field_name.replace("_", " ").title(),
                    "field": f"{matched_section}.{field_name}",
                    "before": _format_scalar_snapshot(before_value),
                    "after": _format_scalar_snapshot(next_value),
                }
            )

    if not changes:
        raise ValueError("Select at least one adjusted field before running the simulator")

    return payload, changes


def _simulate_saved_application(
    runtime: MinimalRuntime,
    application_id: int,
    requested_changes: Mapping[str, Any],
) -> dict[str, Any]:
    application = get_application_by_id(runtime.application_db_path, application_id)
    if application is None:
        raise LookupError(application_id)

    original_payload = _parse_application_payload_json(application)
    expected_tier = _validate_saved_payload(runtime, original_payload)
    simulated_payload, changes = _build_simulation_changes(original_payload, requested_changes)
    simulated_tier = _validate_saved_payload(runtime, simulated_payload)
    if simulated_tier != expected_tier:
        raise RuntimeError(
            f"Simulated payload tier mismatch: expected {expected_tier} but derived {simulated_tier}"
        )

    simulated_response = _score_payload(runtime, simulated_payload)
    original_probability = application.get("last_probability")
    simulated_probability = simulated_response.get("probability_of_default")
    delta = None
    if isinstance(original_probability, (int, float)) and isinstance(simulated_probability, (int, float)):
        delta = float(simulated_probability) - float(original_probability)
    delta_direction = _simulation_delta_direction(delta)
    original_decision = application.get("last_decision")
    simulated_decision = simulated_response.get("decision")

    return {
        "application_id": application_id,
        "coverage_tier": simulated_response.get("coverage_tier", expected_tier),
        "original_probability": original_probability,
        "original_probability_text": _format_probability_pct(original_probability),
        "original_decision": original_decision,
        "original_decision_label": _decision_label(original_decision),
        "simulated_probability": simulated_probability,
        "simulated_probability_text": _format_probability_pct(simulated_probability),
        "simulated_decision": simulated_decision,
        "simulated_decision_label": _decision_label(simulated_decision),
        "simulated_decision_badge_class": _decision_badge_class(simulated_decision),
        "delta": delta,
        "delta_text": _delta_text(delta),
        "delta_direction": delta_direction,
        "decision_delta_label": _decision_delta_label(original_decision, simulated_decision),
        "decision_changed": str(original_decision or "").upper() != str(simulated_decision or "").upper(),
        "risk_movement_label": _simulation_movement_label(delta_direction),
        "changed_features": changes,
        "change_count": len(changes),
        "model_version": simulated_response.get("model_version"),
        "fairness_audit_passed": simulated_response.get("model_fairness_audit_passed"),
        "top_5_explanations": simulated_response.get("top_5_explanations", []),
        "non_persistent": True,
    }


def _decision_badge_class(decision: str | None) -> str:
    normalized = str(decision or "").upper()
    if normalized == "APPROVE":
        return "approve"
    if normalized == "DECLINE":
        return "decline"
    return "review"


def _decision_label(decision: str | None) -> str:
    normalized = str(decision or "").upper()
    return {
        "APPROVE": "Approve",
        "DECLINE": "Decline",
        "REVIEW": "Review",
    }.get(normalized, "Pending")


def _format_probability_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    return "-"


def _format_scalar_snapshot(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return str(value)


def _simulation_delta_direction(delta: float | None) -> str:
    if delta is None or abs(delta) <= 0.0025:
        return "flat"
    return "up" if delta > 0 else "down"


def _simulation_movement_label(delta_direction: str) -> str:
    return {
        "down": "Risk decreased",
        "flat": "Risk nearly unchanged",
        "up": "Risk increased",
    }.get(delta_direction, "Risk movement unavailable")


def _decision_delta_label(original_decision: Any, simulated_decision: Any) -> str:
    original_label = _decision_label(original_decision)
    simulated_label = _decision_label(simulated_decision)
    if original_label == simulated_label:
        return f"{simulated_label} unchanged"
    return f"{original_label} -> {simulated_label}"


def _delta_text(delta: float | None) -> str:
    if delta is None:
        return "-"
    prefix = "+" if delta > 0 else ""
    return f"{prefix}{delta * 100:.1f} pts"


def _build_ui_config(runtime: MinimalRuntime) -> dict[str, Any]:
    config = dict(_build_demo_config(runtime))
    routes = dict(config.get("routes", {}))
    routes.update(
        {
            "home": "/",
            "analyze": "/analyze",
            "status": "/status",
            "analytics": "/analytics",
            "applicationsNew": "/applications/new",
            "analyst": "/analyst",
        }
    )
    config["routes"] = routes
    return config


def _build_health_snapshot(runtime: MinimalRuntime) -> dict[str, Any]:
    return {
        "status": "ok",
        "model_version": runtime.health_model_version,
        "fairness_audit_passed": runtime.model_fairness_audit_passed,
        "coverage_tiers_available": list(runtime.coverage_tiers_available),
    }


def _build_drift_snapshot(db_path: str) -> dict[str, Any]:
    snapshot = compute_probability_drift_snapshot(
        db_path,
        baseline_mean=DRIFT_BASELINE_PD_MEAN,
        lookback=DRIFT_LOOKBACK_WINDOW,
        watch_threshold=DRIFT_WATCH_THRESHOLD,
        alert_threshold=DRIFT_ALERT_THRESHOLD,
        min_sample_size=DRIFT_MIN_SAMPLE_SIZE,
    )
    return {
        **snapshot,
        "baseline_mean_text": _format_probability_pct(snapshot.get("baseline_mean")),
        "live_mean_text": _format_probability_pct(snapshot.get("live_mean")),
        "live_std_text": _format_probability_pct(snapshot.get("live_std")),
        "deviation_pct_text": (
            f"{snapshot['deviation_pct']:+.1f}%"
            if isinstance(snapshot.get("deviation_pct"), (int, float))
            else "—"
        ),
        "status_label": str(snapshot.get("status", "watch")).capitalize(),
        "status_badge_class": {
            "stable": "approve",
            "watch": "review",
            "alert": "decline",
        }.get(str(snapshot.get("status", "watch")), "review"),
    }


def _build_analytics_config(runtime: MinimalRuntime) -> dict[str, Any]:
    return {
        "plots": {section: [] for section in ("eda", "eval", "shap", "fairness")},
        "qualityReport": None,
        "championReport": None,
        "drift": _build_drift_snapshot(runtime.application_db_path),
        "fairnessComparison": {"mode": "offline_only", "mode_label": "Offline comparison"},
        "fairnessTables": {},
    }


def _request_data() -> dict[str, Any]:
    if request.is_json:
        payload = request.get_json(silent=True)
        return payload if isinstance(payload, dict) else {}
    return request.form.to_dict()


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _load_payload_from_input(raw_value: Any) -> dict[str, Any]:
    payload = raw_value
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            raise ValueError("application_payload_json is required")
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("application payload must be a JSON object")
    return payload


def _build_create_application_args(runtime: MinimalRuntime) -> dict[str, Any]:
    data = _request_data()
    applicant_name = str(data.get("applicant_name", "")).strip()
    if not applicant_name:
        raise ValueError("applicant_name is required")

    payload = _load_payload_from_input(
        data.get("application_payload") if "application_payload" in data else data.get("application_payload_json")
    )
    inferred_tier = _validate_saved_payload(runtime, payload)
    supplied_tier = str(data.get("tier_type", "")).strip().upper() or inferred_tier
    if supplied_tier != inferred_tier:
        raise ValueError(f"tier_type does not match payload coverage tier ({inferred_tier})")

    return {
        "applicant_name": applicant_name,
        "sk_id_curr": _coerce_optional_int(data.get("sk_id_curr")),
        "tier_type": supplied_tier,
        "current_status": str(data.get("current_status", "SUBMITTED")).strip().upper() or "SUBMITTED",
        "application_payload_json": payload,
    }


def _build_update_application_args(runtime: MinimalRuntime) -> dict[str, Any]:
    data = _request_data()
    updates: dict[str, Any] = {}

    if "applicant_name" in data:
        updates["applicant_name"] = str(data.get("applicant_name", "")).strip()

    if "sk_id_curr" in data:
        updates["sk_id_curr"] = _coerce_optional_int(data.get("sk_id_curr"))

    payload_supplied = "application_payload" in data or "application_payload_json" in data
    payload: dict[str, Any] | None = None
    inferred_tier: str | None = None
    if payload_supplied:
        payload = _load_payload_from_input(
            data.get("application_payload") if "application_payload" in data else data.get("application_payload_json")
        )
        inferred_tier = _validate_saved_payload(runtime, payload)
        updates["application_payload_json"] = payload

    if "tier_type" in data:
        tier_type = str(data.get("tier_type", "")).strip().upper()
        if tier_type:
            if inferred_tier is not None and tier_type != inferred_tier:
                raise ValueError(f"tier_type does not match payload coverage tier ({inferred_tier})")
            updates["tier_type"] = tier_type
    elif inferred_tier is not None:
        updates["tier_type"] = inferred_tier

    if "current_status" in data:
        status = str(data.get("current_status", "")).strip().upper()
        if status:
            updates["current_status"] = status

    return updates


def _load_application_or_404(runtime: MinimalRuntime, application_id: int) -> dict[str, Any]:
    application = get_application_by_id(runtime.application_db_path, application_id)
    if application is None:
        abort(404)
    return application


def _application_view_model(application: Mapping[str, Any]) -> dict[str, Any]:
    view_model = dict(application)
    try:
        payload_obj = _parse_application_payload_json(application)
    except Exception:
        payload_obj = None
    view_model["application_payload_obj"] = payload_obj
    view_model["application_payload_pretty"] = (
        json.dumps(payload_obj, indent=2)
        if payload_obj is not None
        else application.get("application_payload_json", "")
    )
    view_model["status_label"] = _status_label(application.get("current_status"))
    view_model["status_badge_class"] = _status_badge_class(application.get("current_status"))
    view_model["decision_badge_class"] = _decision_badge_class(application.get("last_decision"))
    view_model["decision_label"] = _decision_label(application.get("last_decision"))
    view_model["last_probability_text"] = _format_probability_pct(application.get("last_probability"))
    return view_model


def _score_run_view_model(score_run: Mapping[str, Any]) -> dict[str, Any]:
    view_model = dict(score_run)
    raw_payload = score_run.get("score_payload_json")
    try:
        payload_obj = json.loads(raw_payload) if isinstance(raw_payload, str) else None
    except Exception:
        payload_obj = None
    view_model["score_payload_obj"] = payload_obj
    view_model["score_payload_pretty"] = json.dumps(payload_obj, indent=2) if payload_obj is not None else raw_payload
    view_model["decision_badge_class"] = _decision_badge_class(score_run.get("decision"))
    view_model["decision_label"] = _decision_label(score_run.get("decision"))
    view_model["probability_text"] = _format_probability_pct(score_run.get("probability"))
    return view_model


def _change_log_view_model(change_log: Mapping[str, Any]) -> dict[str, Any]:
    view_model = dict(change_log)
    raw_summary = change_log.get("change_summary_json")
    try:
        summary_obj = json.loads(raw_summary) if isinstance(raw_summary, str) else None
    except Exception:
        summary_obj = None
    view_model["change_summary_obj"] = summary_obj
    view_model["changes"] = summary_obj.get("changes", []) if isinstance(summary_obj, dict) else []
    return view_model


def _status_label(status: Any) -> str:
    return str(status or "-").replace("_", " ")


def _status_badge_class(status: Any) -> str:
    normalized = str(status or "").upper()
    if normalized == "APPROVED":
        return "approve"
    if normalized == "DECLINED":
        return "decline"
    if normalized in {"REVIEW", "READY_FOR_REVIEW"}:
        return "review"
    return "neutral"


def _status_counts(applications: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(ALLOWED_STATUS_VALUES)}
    for application in applications:
        status = str(application.get("current_status", "")).upper()
        if status in counts:
            counts[status] += 1
    return counts


def _dashboard_stats(applications: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(applications)
    analyzed = 0
    review_queue = 0
    decisioned = 0
    probabilities: list[float] = []

    for application in applications:
        status = str(application.get("current_status", "")).upper()
        if status == "ANALYZED":
            analyzed += 1
        if status in {"SUBMITTED", "READY_FOR_REVIEW", "REVIEW"}:
            review_queue += 1
        if status in {"APPROVED", "REVIEW", "DECLINED"}:
            decisioned += 1

        probability = application.get("last_probability")
        if isinstance(probability, (int, float)):
            probabilities.append(float(probability))

    avg_probability = sum(probabilities) / len(probabilities) if probabilities else None
    return {
        "total": total,
        "review_queue": review_queue,
        "analyzed": analyzed,
        "decisioned": decisioned,
        "avg_probability": avg_probability,
    }


def _build_change_summary(
    before_application: Mapping[str, Any],
    after_application: Mapping[str, Any],
) -> dict[str, Any] | None:
    changes: list[dict[str, Any]] = []

    for field_name in ("applicant_name", "sk_id_curr", "tier_type", "current_status"):
        before_value = before_application.get(field_name)
        after_value = after_application.get(field_name)
        if before_value != after_value:
            changes.append({"field": field_name, "before": before_value, "after": after_value})

    before_payload = before_application.get("application_payload_obj") or {}
    after_payload = after_application.get("application_payload_obj") or {}
    section_names = sorted(set(before_payload) | set(after_payload))
    for section_name in section_names:
        before_section = before_payload.get(section_name, {}) if isinstance(before_payload.get(section_name, {}), dict) else {}
        after_section = after_payload.get(section_name, {}) if isinstance(after_payload.get(section_name, {}), dict) else {}
        for field_name in sorted(set(before_section) | set(after_section)):
            before_value = before_section.get(field_name)
            after_value = after_section.get(field_name)
            if before_value != after_value:
                changes.append(
                    {
                        "field": f"{section_name}.{field_name}",
                        "before": before_value,
                        "after": after_value,
                    }
                )

    if not changes:
        return None

    return {
        "application_id": after_application.get("id"),
        "change_count": len(changes),
        "changes": changes,
    }


def _analyze_saved_application(runtime: MinimalRuntime, application_id: int) -> dict[str, Any]:
    application = _load_application_or_404(runtime, application_id)
    payload = _parse_application_payload_json(application)
    response = _score_payload(runtime, payload)
    save_score_run(
        runtime.application_db_path,
        application_id=application_id,
        probability=response.get("probability_of_default"),
        decision=response.get("decision"),
        model_version=response.get("model_version"),
        score_payload_json=response,
    )
    return response


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _humanize_feature_name(feature_name: str) -> str:
    text = str(feature_name or "").replace("_", " ").strip().lower()
    replacements = {
        "amt": "amount",
        "cnt": "count",
        "dpd": "days past due",
        "ext": "external",
        "src": "source",
        "req": "request",
        "curr": "current",
        "prev": "previous",
        "inst": "installment",
        "pos": "pos",
        "cc": "credit card",
    }
    words = [replacements.get(word, word) for word in text.split()]
    return " ".join(words).strip().capitalize() or "Feature signal"


def _build_shap_narratives(explanations: list[dict[str, Any]]) -> list[str]:
    if not explanations:
        return ["Model explanation artifacts are not available for this score run."]

    narratives: list[str] = []
    ranking_terms = ["strongest", "second-strongest", "next", "additional", "final"]
    for index, item in enumerate(explanations[:5]):
        feature = _humanize_feature_name(str(item.get("feature", "")))
        reason = str(item.get("reason", "")).strip()
        reason = reason[:-1] if reason.endswith(".") else reason
        rank = ranking_terms[index] if index < len(ranking_terms) else "additional"
        if reason:
            sentence = f"{feature} is the {rank} risk driver in this assessment because {reason.lower()}."
        else:
            sentence = f"{feature} is an {rank} model driver in this assessment."
        narratives.append(sentence[0].upper() + sentence[1:])
    return narratives


def _build_application_snapshot(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    application = payload.get("application", {}) if isinstance(payload.get("application"), dict) else {}
    rows = [
        ("Income Total", application.get("AMT_INCOME_TOTAL_CAPPED")),
        ("Credit Amount", application.get("AMT_CREDIT")),
        ("Annuity", application.get("AMT_ANNUITY")),
        ("Goods Price", application.get("AMT_GOODS_PRICE")),
        ("Employment Days", application.get("DAYS_EMPLOYED")),
        ("Age Days", application.get("DAYS_BIRTH")),
    ]
    return [{"label": label, "value": _format_scalar_snapshot(value)} for label, value in rows]


def _build_external_source_snapshot(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    application = payload.get("application", {}) if isinstance(payload.get("application"), dict) else {}
    rows = [
        ("External Source 1", application.get("EXT_SOURCE_1")),
        ("External Source 2", application.get("EXT_SOURCE_2")),
        ("External Source 3", application.get("EXT_SOURCE_3")),
    ]
    return [{"label": label, "value": _format_scalar_snapshot(value)} for label, value in rows]


def _build_aggregate_snapshot(payload: Mapping[str, Any], tier_type: str) -> list[dict[str, str]]:
    if str(tier_type).upper() != "FULL":
        return [{"label": "Aggregate Coverage", "value": "Application-only REDUCED payload"}]

    rows: list[dict[str, str]] = []
    labels = {
        "bureau_agg": "Bureau",
        "previous_agg": "Previous Applications",
        "installments_agg": "Installments",
        "pos_cash_agg": "POS Cash",
        "credit_card_agg": "Credit Card",
    }
    for section_name, label in labels.items():
        section = payload.get(section_name, {}) if isinstance(payload.get(section_name), dict) else {}
        values = list(section.values())
        populated = sum(
            1
            for value in values
            if value not in (None, "") and not (isinstance(value, (int, float)) and float(value) == 0.0)
        )
        rows.append({"label": label, "value": f"{populated}/{len(values)} populated"})
    return rows


def _decision_summary(decision: str | None) -> str:
    normalized = str(decision or "").upper()
    if normalized == "APPROVE":
        return "The calibrated risk signal is inside the approval range for this policy."
    if normalized == "DECLINE":
        return "The calibrated risk signal is outside the acceptable range for this policy."
    if normalized == "REVIEW":
        return "The calibrated risk signal requires analyst review before final action."
    return "This application has not been analyzed yet."


def _format_timestamp_display(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return text.replace("T", " ")
    if parsed.tzinfo is None:
        return parsed.strftime("%d %b %Y, %H:%M")
    return parsed.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC")


def _format_simulator_value(value: Any, field: Mapping[str, Any]) -> str:
    if value is None or value == "":
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    kind = str(field.get("value_kind", "")).lower()
    decimals = int(field.get("decimals", 0 if kind == "amount" else 2))
    if kind == "amount":
        return f"{numeric:,.0f}"
    return f"{numeric:.{decimals}f}"


def _build_simulator_fields(payload: Mapping[str, Any], tier_type: str) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for spec in WHAT_IF_FIELD_SPECS:
        if str(tier_type).upper() != "FULL" and spec["section"] != "application":
            continue
        section = payload.get(spec["section"], {}) if isinstance(payload.get(spec["section"]), dict) else {}
        value = section.get(spec["name"])
        field = dict(spec)
        field["input_id"] = f"sim-{spec['section']}-{spec['name']}".replace("_", "-").lower()
        field["value"] = "" if value is None else value
        field["value_display"] = _format_simulator_value(value, field)
        field["slider_enabled"] = field.get("control") == "slider"
        if field["slider_enabled"]:
            field["min_display"] = _format_simulator_value(field.get("min"), field)
            field["max_display"] = _format_simulator_value(field.get("max"), field)
        fields.append(field)
    return fields


def _build_report_context(
    application: Mapping[str, Any],
    latest_score_run: Mapping[str, Any] | None,
    score_history: list[dict[str, Any]],
    health_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    payload = application.get("application_payload_obj") if isinstance(application.get("application_payload_obj"), dict) else {}
    score_payload = (
        latest_score_run.get("score_payload_obj")
        if latest_score_run and isinstance(latest_score_run.get("score_payload_obj"), dict)
        else {}
    )
    explanations = score_payload.get("top_5_explanations")
    if not isinstance(explanations, list):
        explanations = []

    probability = _first_present(
        score_payload.get("probability_of_default"),
        latest_score_run.get("probability") if latest_score_run else None,
        application.get("last_probability"),
    )
    decision = _first_present(
        score_payload.get("decision"),
        latest_score_run.get("decision") if latest_score_run else None,
        application.get("last_decision"),
    )
    model_version = str(
        _first_present(
            score_payload.get("model_version"),
            latest_score_run.get("model_version") if latest_score_run else None,
            application.get("last_model_version"),
            health_snapshot.get("model_version"),
        )
        or "-"
    )
    coverage_tier = str(_first_present(score_payload.get("coverage_tier"), application.get("tier_type")) or "-").upper()
    fairness_audit_passed = bool(
        _first_present(
            score_payload.get("model_fairness_audit_passed"),
            health_snapshot.get("fairness_audit_passed"),
        )
    )
    analyzed_at = latest_score_run.get("scored_at") if latest_score_run else None
    gauge_value = 0.0
    if isinstance(probability, (int, float)):
        gauge_value = min(max(float(probability) * 100.0, 0.0), 100.0)

    top_drivers = [
        {
            "rank": index + 1,
            "feature_label": _humanize_feature_name(str(item.get("feature", ""))),
            "reason": str(item.get("reason", "")).strip() or "No textual explanation available for this feature.",
            "weight_pct": max(44, 100 - index * 14),
        }
        for index, item in enumerate(explanations[:5])
    ]

    history_items: list[dict[str, Any]] = []
    for index, run in enumerate(score_history):
        run_payload = run.get("score_payload_obj") if isinstance(run.get("score_payload_obj"), dict) else {}
        run_probability = _first_present(run.get("probability"), run_payload.get("probability_of_default"))
        run_decision = _first_present(run.get("decision"), run_payload.get("decision"))
        history_items.append(
            {
                "id": run.get("id"),
                "is_latest": index == 0,
                "scored_at": run.get("scored_at"),
                "scored_at_display": _format_timestamp_display(run.get("scored_at")),
                "probability_text": _format_probability_pct(run_probability),
                "decision": run_decision,
                "decision_label": _decision_label(run_decision),
                "decision_badge_class": _decision_badge_class(run_decision),
                "model_version": str(_first_present(run.get("model_version"), run_payload.get("model_version")) or "-"),
            }
        )

    last_model_result = [
        {"label": "Current Status", "value": str(application.get("current_status") or "-").replace("_", " ")},
        {"label": "Decision", "value": _decision_label(decision)},
        {"label": "Calibrated PD", "value": _format_probability_pct(probability)},
        {"label": "Model Version", "value": model_version},
    ]

    return {
        "has_score_run": latest_score_run is not None,
        "decision": decision,
        "decision_label": _decision_label(decision),
        "decision_badge_class": _decision_badge_class(decision),
        "decision_summary": _decision_summary(decision),
        "probability_value": float(probability) if isinstance(probability, (int, float)) else None,
        "probability_text": _format_probability_pct(probability),
        "gauge_value": gauge_value,
        "model_version": model_version,
        "coverage_tier": coverage_tier,
        "fairness_audit_passed": fairness_audit_passed,
        "analyzed_at_display": _format_timestamp_display(analyzed_at),
        "submission_display": _format_timestamp_display(application.get("submitted_at")),
        "updated_display": _format_timestamp_display(application.get("updated_at")),
        "shap_narratives": _build_shap_narratives(explanations),
        "top_drivers": top_drivers,
        "application_snapshot": _build_application_snapshot(payload),
        "external_sources": _build_external_source_snapshot(payload),
        "aggregate_snapshot": _build_aggregate_snapshot(payload, coverage_tier),
        "adverse_action": {
            "section_title": "Adverse Action Style Summary",
            "summary": "Top model drivers and calibrated probability are shown for analyst review.",
            "reasons": [],
            "emphasize_adverse": str(decision or "").upper() == "DECLINE",
        },
        "last_model_result": last_model_result,
        "score_history": history_items,
        "score_history_count": len(history_items),
        "raw_output_pretty": latest_score_run.get("score_payload_pretty") if latest_score_run else "",
        "simulator_fields": _build_simulator_fields(payload, coverage_tier),
    }


def _build_report_chat_context(
    application: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    top_drivers = report.get("top_drivers") if isinstance(report.get("top_drivers"), list) else []
    score_history = report.get("score_history") if isinstance(report.get("score_history"), list) else []

    return {
        "scope": "analyst_application_report",
        "application_id": application.get("id"),
        "applicant_summary": {
            "applicant_name": application.get("applicant_name"),
            "sk_id_curr": application.get("sk_id_curr"),
            "coverage_tier": report.get("coverage_tier"),
            "current_status": str(application.get("current_status") or "-").replace("_", " "),
            "submitted_at": report.get("submission_display"),
            "updated_at": report.get("updated_display"),
        },
        "latest_assessment": {
            "decision": report.get("decision"),
            "decision_label": report.get("decision_label"),
            "calibrated_probability": report.get("probability_value"),
            "calibrated_probability_text": report.get("probability_text"),
            "decision_summary": report.get("decision_summary"),
            "model_version": report.get("model_version"),
            "fairness_audit_passed": report.get("fairness_audit_passed"),
            "analyzed_at": report.get("analyzed_at_display"),
        },
        "top_drivers": [
            {
                "feature": item.get("feature_label"),
                "reason": item.get("reason"),
                "rank": item.get("rank"),
            }
            for item in top_drivers[:5]
        ],
        "shap_narratives": [
            str(item).strip()
            for item in report.get("shap_narratives", [])[:5]
            if str(item).strip()
        ],
        "score_history": [
            {
                "scored_at": item.get("scored_at_display"),
                "decision": item.get("decision"),
                "decision_label": item.get("decision_label"),
                "probability_text": item.get("probability_text"),
                "model_version": item.get("model_version"),
                "is_latest": bool(item.get("is_latest")),
            }
            for item in score_history[:5]
        ],
        "adverse_action": {
            "section_title": "Adverse Action Style Summary",
            "summary": "Top model drivers and calibrated probability are shown for analyst review.",
            "reasons": [],
        },
        "simulator_result": None,
        "analyst_guidance": {
            "disclaimer": "Analyst assistant only. Responses are grounded to this application report and are not automated lending decisions."
        },
    }


if __name__ == "__main__":
    app = create_app()
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", os.environ.get("FLASK_RUN_PORT", "5050"))),
    )
