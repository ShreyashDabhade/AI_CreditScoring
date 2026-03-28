from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from jinja2 import ChoiceLoader, FileSystemLoader
import pandas as pd

from src.api import determine_coverage_tier, score_request, validate_payload
from src.api.app import RUNTIME_EXTENSION_KEY, _build_demo_config, _build_demo_seed_payload, create_app
from src.db_manager import (
    ALLOWED_STATUS_VALUES,
    create_application,
    get_application_by_id,
    list_application_change_history,
    list_applications,
    list_score_history_for_application,
    save_application_change_log,
    save_score_run,
    update_application,
)


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
WHAT_IF_FIELD_SPECS = (
    {
        "section": "application",
        "name": "AMT_CREDIT",
        "label": "Credit Amount",
        "step": "1000",
        "min": "0",
    },
    {
        "section": "application",
        "name": "AMT_ANNUITY",
        "label": "Annuity",
        "step": "500",
        "min": "0",
    },
    {
        "section": "application",
        "name": "AMT_INCOME_TOTAL_CAPPED",
        "label": "Income Total",
        "step": "1000",
        "min": "0",
    },
    {
        "section": "application",
        "name": "EXT_SOURCE_2",
        "label": "External Source 2",
        "step": "0.01",
        "min": "0",
        "max": "1",
    },
    {
        "section": "application",
        "name": "EXT_SOURCE_3",
        "label": "External Source 3",
        "step": "0.01",
        "min": "0",
        "max": "1",
    },
    {
        "section": "bureau_agg",
        "name": "BUREAU_DEBT_TO_CREDIT_RATIO",
        "label": "Bureau Debt / Credit Ratio",
        "step": "0.01",
        "min": "0",
    },
    {
        "section": "previous_agg",
        "name": "PREV_APPROVAL_RATE",
        "label": "Previous Approval Rate",
        "step": "0.01",
        "min": "0",
        "max": "1",
    },
    {
        "section": "installments_agg",
        "name": "INST_DPD_MEAN",
        "label": "Installment DPD Mean",
        "step": "1",
        "min": "0",
    },
    {
        "section": "pos_cash_agg",
        "name": "POS_DPD_MEAN",
        "label": "POS Cash DPD Mean",
        "step": "0.5",
        "min": "0",
    },
    {
        "section": "credit_card_agg",
        "name": "CC_UTILIZATION_MEAN",
        "label": "Credit Card Utilization Mean",
        "step": "0.01",
        "min": "0",
    },
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
            "applicationsNew": "/applications/new",
            "analyst": "/analyst",
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


def _render_command_center_page(
    app: Flask,
    template_name: str,
    *,
    page_title: str,
    active_nav: str,
    **context: Any,
):
    return render_template(
        template_name,
        page_title=page_title,
        active_nav=active_nav,
        ui_config=_build_ui_config(app),
        health_snapshot=_build_health_snapshot(app),
        allowed_statuses=sorted(ALLOWED_STATUS_VALUES),
        **context,
    )


def _application_db_path(app: Flask) -> str:
    db_path = app.config.get("APPLICATION_DB_PATH")
    if not isinstance(db_path, str) or not db_path:
        raise RuntimeError("Application database path is not configured")
    return db_path


def _wants_json_response() -> bool:
    if request.args.get("format", "").strip().lower() == "json":
        return True
    if request.is_json:
        return True
    best = request.accept_mimetypes.best_match(["application/json", "text/html"])
    return best == "application/json" and (
        request.accept_mimetypes["application/json"]
        >= request.accept_mimetypes["text/html"]
    )


def _request_data() -> dict[str, Any]:
    if request.is_json:
        payload = request.get_json(silent=True)
        if isinstance(payload, dict):
            return payload
        return {}
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


def _validate_saved_payload(payload: dict[str, Any]) -> str:
    error_code, missing_fields = validate_payload(payload)
    if error_code is not None:
        message = f"application payload is not scoreable: {error_code}"
        if missing_fields:
            message = f"{message} ({', '.join(missing_fields)})"
        raise ValueError(message)
    return determine_coverage_tier(payload)


def _build_create_application_args() -> dict[str, Any]:
    data = _request_data()
    applicant_name = str(data.get("applicant_name", "")).strip()
    if not applicant_name:
        raise ValueError("applicant_name is required")

    payload = _load_payload_from_input(
        data.get("application_payload") if "application_payload" in data else data.get("application_payload_json")
    )
    inferred_tier = _validate_saved_payload(payload)
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


def _build_update_application_args() -> dict[str, Any]:
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
        inferred_tier = _validate_saved_payload(payload)
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


def _load_application_or_404(app: Flask, application_id: int) -> dict[str, Any]:
    application = get_application_by_id(_application_db_path(app), application_id)
    if application is None:
        abort(404)
    return application


def _parse_application_payload_json(application: dict[str, Any]) -> dict[str, Any]:
    raw_payload = application.get("application_payload_json")
    if not isinstance(raw_payload, str):
        raise ValueError("stored application payload is missing")
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise ValueError("stored application payload must be a JSON object")
    return payload


def _status_label(status: str | None) -> str:
    return str(status or "-").replace("_", " ")


def _status_badge_class(status: str | None) -> str:
    normalized = str(status or "").upper()
    if normalized == "APPROVED":
        return "approve"
    if normalized == "DECLINED":
        return "decline"
    if normalized in {"REVIEW", "READY_FOR_REVIEW"}:
        return "review"
    return "neutral"


def _application_view_model(application: dict[str, Any]) -> dict[str, Any]:
    view_model = dict(application)
    try:
        payload_obj = _parse_application_payload_json(application)
    except Exception:
        payload_obj = None
    view_model["application_payload_obj"] = payload_obj
    view_model["application_payload_pretty"] = json.dumps(payload_obj, indent=2) if payload_obj is not None else application.get("application_payload_json", "")
    view_model["status_label"] = _status_label(application.get("current_status"))
    view_model["status_badge_class"] = _status_badge_class(application.get("current_status"))
    view_model["decision_badge_class"] = _decision_badge_class(application.get("last_decision"))
    view_model["decision_label"] = _decision_label(application.get("last_decision"))
    view_model["last_probability_text"] = _format_probability_pct(application.get("last_probability"))
    return view_model


def _score_run_view_model(score_run: dict[str, Any]) -> dict[str, Any]:
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


def _change_log_view_model(change_log: dict[str, Any]) -> dict[str, Any]:
    view_model = dict(change_log)
    raw_summary = change_log.get("change_summary_json")
    try:
        summary_obj = json.loads(raw_summary) if isinstance(raw_summary, str) else None
    except Exception:
        summary_obj = None
    view_model["change_summary_obj"] = summary_obj
    view_model["changes"] = summary_obj.get("changes", []) if isinstance(summary_obj, dict) else []
    return view_model


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


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _format_probability_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    return "—"


def _format_scalar_snapshot(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return str(value)


def _build_application_snapshot(payload: dict[str, Any]) -> list[dict[str, str]]:
    application = payload.get("application", {}) if isinstance(payload.get("application"), dict) else {}
    rows = [
        ("Income Total", application.get("AMT_INCOME_TOTAL_CAPPED")),
        ("Credit Amount", application.get("AMT_CREDIT")),
        ("Annuity", application.get("AMT_ANNUITY")),
        ("Goods Price", application.get("AMT_GOODS_PRICE")),
        ("External Source 1", application.get("EXT_SOURCE_1")),
        ("External Source 2", application.get("EXT_SOURCE_2")),
        ("External Source 3", application.get("EXT_SOURCE_3")),
        ("Employment Days", application.get("DAYS_EMPLOYED")),
    ]
    return [{"label": label, "value": _format_scalar_snapshot(value)} for label, value in rows]


def _build_aggregate_snapshot(payload: dict[str, Any], tier_type: str) -> list[dict[str, str]]:
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


def _decision_badge_class(decision: str | None) -> str:
    normalized = str(decision or "").upper()
    if normalized == "APPROVE":
        return "approve"
    if normalized == "DECLINE":
        return "decline"
    return "review"


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


def _build_application_snapshot(payload: dict[str, Any]) -> list[dict[str, str]]:
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


def _build_external_source_snapshot(payload: dict[str, Any]) -> list[dict[str, str]]:
    application = payload.get("application", {}) if isinstance(payload.get("application"), dict) else {}
    rows = [
        ("External Source 1", application.get("EXT_SOURCE_1")),
        ("External Source 2", application.get("EXT_SOURCE_2")),
        ("External Source 3", application.get("EXT_SOURCE_3")),
    ]
    return [{"label": label, "value": _format_scalar_snapshot(value)} for label, value in rows]


def _decision_label(decision: str | None) -> str:
    normalized = str(decision or "").upper()
    return {
        "APPROVE": "Approve",
        "DECLINE": "Decline",
        "REVIEW": "Review",
    }.get(normalized, "Pending")


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


def _build_report_context(
    application: dict[str, Any],
    latest_score_run: dict[str, Any] | None,
    score_history: list[dict[str, Any]],
    health_snapshot: dict[str, Any],
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
        "last_model_result": last_model_result,
        "score_history": history_items,
        "score_history_count": len(history_items),
        "raw_output_pretty": latest_score_run.get("score_payload_pretty") if latest_score_run else "",
        "simulator_fields": _build_simulator_fields(payload, coverage_tier),
    }


def _clone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _build_simulator_fields(payload: dict[str, Any], tier_type: str) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for spec in WHAT_IF_FIELD_SPECS:
        if str(tier_type).upper() != "FULL" and spec["section"] != "application":
            continue
        section = payload.get(spec["section"], {}) if isinstance(payload.get(spec["section"]), dict) else {}
        value = section.get(spec["name"])
        field = dict(spec)
        field["input_id"] = f"sim-{spec['section']}-{spec['name']}".replace("_", "-").lower()
        field["value"] = "" if value is None else value
        field["value_display"] = _format_scalar_snapshot(value)
        fields.append(field)
    return fields


def _coerce_simulator_numeric_value(raw_value: Any, field: dict[str, Any]) -> float:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError(f"{field['label']} is required for simulation")
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{field['label']} must be numeric") from exc

    min_value = field.get("min")
    max_value = field.get("max")
    if min_value is not None and value < float(min_value):
        raise ValueError(f"{field['label']} must be at least {min_value}")
    if max_value is not None and value > float(max_value):
        raise ValueError(f"{field['label']} must be at most {max_value}")
    return value


def _build_simulation_changes(
    original_payload: dict[str, Any],
    requested_changes: dict[str, Any],
    tier_type: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _clone_payload(original_payload)
    changes: list[dict[str, Any]] = []

    for field in _build_simulator_fields(original_payload, tier_type):
        if field["name"] not in requested_changes:
            continue
        next_value = _coerce_simulator_numeric_value(requested_changes.get(field["name"]), field)
        section_name = field["section"]
        section = payload.setdefault(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{field['label']} cannot be simulated for this payload")
        before_value = section.get(field["name"])
        section[field["name"]] = next_value
        if before_value != next_value:
            changes.append(
                {
                    "label": field["label"],
                    "field": f"{section_name}.{field['name']}",
                    "before": _format_scalar_snapshot(before_value),
                    "after": _format_scalar_snapshot(next_value),
                }
            )

    if not changes:
        raise ValueError("Select at least one adjusted field before running the simulator")

    return payload, changes


def _delta_text(delta: float | None) -> str:
    if delta is None:
        return "-"
    prefix = "+" if delta > 0 else ""
    return f"{prefix}{delta * 100:.1f} pts"


def _simulate_saved_application(
    app: Flask,
    application_id: int,
    requested_changes: dict[str, Any],
) -> dict[str, Any]:
    application = _application_view_model(_load_application_or_404(app, application_id))
    original_payload = _parse_application_payload_json(application)
    expected_tier = _validate_saved_payload(original_payload)
    simulated_payload, changes = _build_simulation_changes(original_payload, requested_changes, expected_tier)
    simulated_tier = _validate_saved_payload(simulated_payload)
    if simulated_tier != expected_tier:
        raise RuntimeError(
            f"Simulated payload tier mismatch: expected {expected_tier} but derived {simulated_tier}"
        )

    runtime = _runtime(app)
    simulated_response = score_request(
        simulated_payload,
        runtime,
        mock_mode=bool(getattr(runtime, "mock_mode", False)),
    )
    original_probability = application.get("last_probability")
    simulated_probability = simulated_response.get("probability_of_default")
    delta = None
    if isinstance(original_probability, (int, float)) and isinstance(simulated_probability, (int, float)):
        delta = float(simulated_probability) - float(original_probability)

    return {
        "application_id": application_id,
        "coverage_tier": simulated_response.get("coverage_tier", expected_tier),
        "original_probability": original_probability,
        "original_probability_text": _format_probability_pct(original_probability),
        "original_decision": application.get("last_decision"),
        "original_decision_label": _decision_label(application.get("last_decision")),
        "simulated_probability": simulated_probability,
        "simulated_probability_text": _format_probability_pct(simulated_probability),
        "simulated_decision": simulated_response.get("decision"),
        "simulated_decision_label": _decision_label(simulated_response.get("decision")),
        "simulated_decision_badge_class": _decision_badge_class(simulated_response.get("decision")),
        "delta": delta,
        "delta_text": _delta_text(delta),
        "delta_direction": "up" if isinstance(delta, float) and delta > 0 else ("down" if isinstance(delta, float) and delta < 0 else "flat"),
        "changed_features": changes,
        "change_count": len(changes),
        "model_version": simulated_response.get("model_version"),
        "fairness_audit_passed": simulated_response.get("model_fairness_audit_passed"),
        "top_5_explanations": simulated_response.get("top_5_explanations", []),
        "non_persistent": True,
    }


def _seed_demo_applications_if_empty(app: Flask) -> None:
    if app.testing:
        return
    if os.getenv("PYTEST_CURRENT_TEST"):
        return
    if _env_flag("MASTERMIND_ENABLE_DEMO_SEED") is False:
        return

    db_path = _application_db_path(app)
    if list_applications(db_path, limit=1):
        return

    seed_payload = _build_demo_seed_payload()
    seed_rows = [
        {
            "applicant_name": "Avery Cole",
            "sk_id_curr": 910001001,
            "tier_type": "REDUCED",
            "current_status": "SUBMITTED",
            "application_payload_json": {"application": seed_payload["application"]},
        },
        {
            "applicant_name": "Riley Morgan",
            "sk_id_curr": 910001002,
            "tier_type": "FULL",
            "current_status": "READY_FOR_REVIEW",
            "application_payload_json": seed_payload,
        },
        {
            "applicant_name": "Jordan Blake",
            "sk_id_curr": 910001003,
            "tier_type": "REDUCED",
            "current_status": "READY_FOR_REVIEW",
            "application_payload_json": {
                "application": {
                    **seed_payload["application"],
                    "AMT_CREDIT": 310000.0,
                    "AMT_ANNUITY": 29000.0,
                    "EXT_SOURCE_2": 0.31,
                }
            },
        },
        {
            "applicant_name": "Taylor Shah",
            "sk_id_curr": 910001004,
            "tier_type": "FULL",
            "current_status": "READY_FOR_REVIEW",
            "application_payload_json": {
                **seed_payload,
                "application": {
                    **seed_payload["application"],
                    "AMT_INCOME_TOTAL_CAPPED": 150000.0,
                    "EXT_SOURCE_3": 0.72,
                },
            },
        },
    ]

    created_rows = [create_application(db_path, **row) for row in seed_rows]

    for seeded_id in (created_rows[1]["id"], created_rows[3]["id"]):
        try:
            _analyze_saved_application(app, seeded_id)
        except Exception:
            app.logger.exception("Demo seed analysis failed for application id=%s", seeded_id)


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
    before_application: dict[str, Any],
    after_application: dict[str, Any],
) -> dict[str, Any] | None:
    changes: list[dict[str, Any]] = []

    for field_name in ("applicant_name", "sk_id_curr", "tier_type", "current_status"):
        before_value = before_application.get(field_name)
        after_value = after_application.get(field_name)
        if before_value != after_value:
            changes.append(
                {
                    "field": field_name,
                    "before": before_value,
                    "after": after_value,
                }
            )

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


def _analyze_saved_application(app: Flask, application_id: int) -> dict[str, Any]:
    application = _load_application_or_404(app, application_id)
    payload = _parse_application_payload_json(application)
    expected_tier = _validate_saved_payload(payload)

    runtime = _runtime(app)
    response = score_request(payload, runtime, mock_mode=bool(getattr(runtime, "mock_mode", False)))
    response_tier = str(response.get("coverage_tier", "")).upper()
    if response_tier and response_tier != expected_tier:
        raise RuntimeError(
            f"Saved application tier mismatch: expected {expected_tier} but scoring returned {response_tier}"
        )
    save_score_run(
        _application_db_path(app),
        application_id=application_id,
        probability=response.get("probability_of_default"),
        decision=response.get("decision"),
        model_version=response.get("model_version"),
        score_payload_json=response,
    )
    return response


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

    def application_new():
        return _render_command_center_page(
            app,
            "applications_new.html",
            page_title="New Application",
            active_nav="applications",
        )

    def application_create():
        try:
            created = create_application(_application_db_path(app), **_build_create_application_args())
        except ValueError as exc:
            if _wants_json_response():
                return jsonify({"error": str(exc)}), 400
            return _render_command_center_page(
                app,
                "applications_new.html",
                page_title="New Application",
                active_nav="applications",
                error_message=str(exc),
            ), 400
        return redirect(url_for("application_detail", application_id=created["id"], created=1))

    def application_detail(application_id: int):
        application = _application_view_model(_load_application_or_404(app, application_id))
        return _render_command_center_page(
            app,
            "application_detail.html",
            page_title=f"Application #{application_id}",
            active_nav="applications",
            application=application,
            score_history=[_score_run_view_model(item) for item in list_score_history_for_application(_application_db_path(app), application_id)],
            detail_mode="applicant",
        )

    def analyst_dashboard():
        applications = [_application_view_model(item) for item in list_applications(_application_db_path(app))]
        selected_id = request.args.get("selected", type=int)
        selected_application = None
        if applications:
            if selected_id is not None:
                selected_application = next((item for item in applications if item["id"] == selected_id), None)
            if selected_application is None:
                selected_application = applications[0]
        return _render_command_center_page(
            app,
            "analyst_dashboard.html",
            page_title="Analyst Command Center",
            active_nav="analyst",
            applications=applications,
            status_counts=_status_counts(applications),
            dashboard_stats=_dashboard_stats(applications),
            selected_application=selected_application,
        )

    def analyst_applications():
        applications = [_application_view_model(item) for item in list_applications(_application_db_path(app))]
        if _wants_json_response():
            return jsonify({"applications": applications, "health_snapshot": _build_health_snapshot(app)})
        return _render_command_center_page(
            app,
            "analyst_applications.html",
            page_title="Analyst Applications",
            active_nav="analyst",
            applications=applications,
            status_counts=_status_counts(applications),
        )

    def analyst_application_detail(application_id: int):
        application = _application_view_model(_load_application_or_404(app, application_id))
        return _render_command_center_page(
            app,
            "analyst_application_detail.html",
            page_title=f"Edit Application #{application_id}",
            active_nav="analyst",
            application=application,
            score_history=[_score_run_view_model(item) for item in list_score_history_for_application(_application_db_path(app), application_id)],
            change_history=[_change_log_view_model(item) for item in list_application_change_history(_application_db_path(app), application_id, limit=8)],
            detail_mode="analyst",
        )

    def analyst_application_update(application_id: int):
        original_application = _application_view_model(_load_application_or_404(app, application_id))
        try:
            updated = update_application(_application_db_path(app), application_id, **_build_update_application_args())
        except ValueError as exc:
            if _wants_json_response():
                return jsonify({"error": str(exc)}), 400
            return redirect(url_for("analyst_application_detail", application_id=application_id, error=str(exc)))

        if updated is None:
            abort(404)
        updated_view = _application_view_model(updated)
        if change_summary := _build_change_summary(original_application, updated_view):
            save_application_change_log(
                _application_db_path(app),
                application_id=application_id,
                change_summary_json=change_summary,
            )
        return redirect(url_for("analyst_application_detail", application_id=application_id))

    def analyst_application_analyze(application_id: int):
        try:
            _analyze_saved_application(app, application_id)
        except ValueError as exc:
            if _wants_json_response():
                return jsonify({"error": str(exc)}), 400
            return redirect(url_for("analyst_application_detail", application_id=application_id, error=str(exc)))
        except Exception as exc:
            app.logger.exception("Persisted application analysis failed for id=%s", application_id)
            if _wants_json_response():
                return jsonify({"error": "analysis_failed", "message": str(exc)}), 500
            return redirect(
                url_for(
                    "analyst_application_detail",
                    application_id=application_id,
                    error="Analysis failed. Review the saved payload and runtime artifacts, then try again.",
                )
            )
        return redirect(url_for("analyst_application_report", application_id=application_id))

    def analyst_application_simulate(application_id: int):
        data = _request_data()
        requested_changes = data.get("changes", data) if isinstance(data, dict) else {}
        if not isinstance(requested_changes, dict):
            return jsonify({"error": "bad_request", "message": "Simulation changes must be an object."}), 400
        try:
            result = _simulate_saved_application(app, application_id, requested_changes)
        except ValueError as exc:
            return jsonify({"error": "validation_error", "message": str(exc)}), 400
        except Exception as exc:
            app.logger.exception("What-if simulation failed for id=%s", application_id)
            return jsonify({"error": "simulation_failed", "message": str(exc)}), 500
        return jsonify(result)

    def analyst_application_report(application_id: int):
        application = _application_view_model(_load_application_or_404(app, application_id))
        score_history = [
            _score_run_view_model(item)
            for item in list_score_history_for_application(_application_db_path(app), application_id)
        ]
        latest_score_run = score_history[0] if score_history else None
        report = _build_report_context(
            application,
            latest_score_run,
            score_history,
            _build_health_snapshot(app),
        )
        return _render_command_center_page(
            app,
            "analyst_application_report.html",
            page_title=f"Application Report #{application_id}",
            active_nav="analyst",
            application=application,
            latest_score_run=latest_score_run,
            score_history=score_history,
            report=report,
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
    app.add_url_rule("/applications/new", endpoint="application_new", view_func=application_new)
    app.add_url_rule("/applications", endpoint="application_create", view_func=application_create, methods=["POST"])
    app.add_url_rule("/applications/<int:application_id>", endpoint="application_detail", view_func=application_detail)
    app.add_url_rule("/analyst", endpoint="analyst_dashboard", view_func=analyst_dashboard)
    app.add_url_rule("/analyst/applications", endpoint="analyst_applications", view_func=analyst_applications)
    app.add_url_rule(
        "/analyst/applications/<int:application_id>",
        endpoint="analyst_application_detail",
        view_func=analyst_application_detail,
    )
    app.add_url_rule(
        "/analyst/applications/<int:application_id>/update",
        endpoint="analyst_application_update",
        view_func=analyst_application_update,
        methods=["POST"],
    )
    app.add_url_rule(
        "/analyst/applications/<int:application_id>/analyze",
        endpoint="analyst_application_analyze",
        view_func=analyst_application_analyze,
        methods=["POST"],
    )
    app.add_url_rule(
        "/analyst/applications/<int:application_id>/simulate",
        endpoint="analyst_application_simulate",
        view_func=analyst_application_simulate,
        methods=["POST"],
    )
    app.add_url_rule(
        "/analyst/applications/<int:application_id>/report",
        endpoint="analyst_application_report",
        view_func=analyst_application_report,
    )
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
_seed_demo_applications_if_empty(app)


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    app.run(host=host, port=port)
