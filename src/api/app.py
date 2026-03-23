"""Module 5 - strict Flask scoring API."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import json
import logging
import os
import re
import tempfile
import time
from types import MappingProxyType, ModuleType
from typing import Any, Mapping
import uuid

from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np
import pandas as pd

from configs.config import (
    APPROVE_THRESHOLD,
    ARTIFACT_DIR,
    DATA_DIR,
    DECLINE_THRESHOLD,
    FAIRNESS_AUDIT_VERSION,
    FULL_REQUIRED_SECTIONS,
    MODEL_VERSIONS,
)
from src.explainability import render_reason, top_5_explanations_from_shap
from src.models.train import decision_from_pd as trained_decision_from_pd, load_artifacts

ROUTER_VERSION = "router_v1.0.0"
POLICY_VERSION = "policy_v1.0.0"
DEFAULT_FAIRNESS_VERSION_TAG = "2026Q1"
RUNTIME_EXTENSION_KEY = "mastermind_runtime"
PROCESSED_MANIFEST_FILENAME = "processed_artifact_manifest.json"
REPRODUCIBILITY_REPORT_FILENAME = "reproducibility_report.json"
FAIRNESS_RESULT_FILENAME = "model_fairness_audit_passed.joblib"
DEMO_DEFAULT_TIER = "REDUCED"
CATEGORICAL_APPLICATION_FIELDS = frozenset(
    {
        "NAME_CONTRACT_TYPE",
        "NAME_TYPE_SUITE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
    }
)
DEMO_FIELD_OPTIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "NAME_CONTRACT_TYPE": ("Cash loans", "Revolving loans"),
        "NAME_TYPE_SUITE": ("Unaccompanied", "Family", "Spouse, partner"),
        "NAME_EDUCATION_TYPE": (
            "Higher education",
            "Secondary / secondary special",
            "Incomplete higher",
        ),
        "NAME_FAMILY_STATUS": ("Married", "Single / not married", "Civil marriage"),
        "OCCUPATION_TYPE": ("Laborers", "Core staff", "Sales staff"),
        "ORGANIZATION_TYPE": (
            "Business Entity Type 3",
            "Self-employed",
            "School",
        ),
        "WEEKDAY_APPR_PROCESS_START": (
            "MONDAY",
            "TUESDAY",
            "WEDNESDAY",
            "THURSDAY",
            "FRIDAY",
        ),
    }
)

# Frozen public API contract for request payloads.
APPLICATION_REQUIRED_FIELDS: tuple[str, ...] = (
    "AMT_INCOME_TOTAL_CAPPED",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "REGION_POPULATION_RELATIVE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CNT_FAM_MEMBERS",
    "OWN_CAR_AGE",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "NAME_CONTRACT_TYPE",
    "NAME_TYPE_SUITE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "DAYS_EMPLOYED_ANOM",
)
AGG_REQUIRED_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "bureau_agg": (
            "BUREAU_LOAN_COUNT",
            "BUREAU_ACTIVE_COUNT",
            "BUREAU_CLOSED_COUNT",
            "BUREAU_AMT_CREDIT_SUM_SUM",
            "BUREAU_AMT_CREDIT_SUM_DEBT_SUM",
            "BUREAU_DEBT_TO_CREDIT_RATIO",
            "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM",
            "BUREAU_CREDIT_DAY_OVERDUE_MAX",
            "BUREAU_DAYS_CREDIT_MAX",
            "BUREAU_CNT_CREDIT_PROLONG_SUM",
        ),
        "previous_agg": (
            "PREV_APP_COUNT",
            "PREV_APPROVED_COUNT",
            "PREV_REFUSED_COUNT",
            "PREV_APPROVAL_RATE",
            "PREV_REFUSAL_RATE",
            "PREV_AMT_APPLICATION_MEAN",
            "PREV_AMT_CREDIT_MEAN",
            "PREV_AMT_GOODS_PRICE_MEAN",
            "PREV_APP_CREDIT_DIFF_MEAN",
            "PREV_DAYS_DECISION_MAX",
            "PREV_RATE_DOWN_PAYMENT_MEAN",
        ),
        "installments_agg": (
            "INST_RECORD_COUNT",
            "INST_MISSED_RATE",
            "INST_DPD_MEAN",
            "INST_DPD_MAX",
            "INST_PAYMENT_RATIO_MEAN",
            "INST_PAYMENT_RATIO_MIN",
            "INST_LATE_COUNT",
        ),
        "pos_cash_agg": (
            "POS_RECORD_COUNT",
            "POS_DPD_MEAN",
            "POS_DPD_MAX",
            "POS_DPD_DEF_MEAN",
            "POS_DPD_DEF_MAX",
            "POS_COMPLETED_RATE",
            "POS_ACTIVE_RATE",
            "POS_CNT_INSTALMENT_FUTURE_MEAN",
        ),
        "credit_card_agg": (
            "CC_RECORD_COUNT",
            "CC_BALANCE_MEAN",
            "CC_LIMIT_MEAN",
            "CC_UTILIZATION_MEAN",
            "CC_PAYMENT_RATIO_MEAN",
            "CC_DPD_MEAN",
            "CC_DPD_MAX",
            "CC_DRAWINGS_ATM_SUM",
            "CC_DRAWINGS_CURRENT_SUM",
        ),
    }
)
ALLOWED_TOP_LEVEL_KEYS = frozenset(FULL_REQUIRED_SECTIONS)
FULL_SECTION_ORDER: tuple[str, ...] = (
    "application",
    "bureau_agg",
    "previous_agg",
    "installments_agg",
    "pos_cash_agg",
    "credit_card_agg",
)
FULL_ONLY_SECTION_ORDER: tuple[str, ...] = FULL_SECTION_ORDER[1:]
FULL_ONLY_SECTIONS = frozenset(FULL_ONLY_SECTION_ORDER)
NUMERIC_APPLICATION_FIELDS = tuple(
    field for field in APPLICATION_REQUIRED_FIELDS if field not in CATEGORICAL_APPLICATION_FIELDS
)
NUMERIC_FIELDS_BY_SECTION: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "application": NUMERIC_APPLICATION_FIELDS,
        **{section_name: fields for section_name, fields in AGG_REQUIRED_FIELDS.items()},
    }
)

ERROR_MESSAGES: Mapping[str, str] = MappingProxyType(
    {
        "bad_request": "Bad request.",
        "missing_application": "Application section is required.",
        "forbidden_field_code_gender": "CODE_GENDER is not allowed.",
        "partial_full_payload_not_allowed": "Partial FULL payloads are not allowed.",
        "missing_application_fields": "Application payload is missing required fields.",
        "missing_aggregate_fields": "FULL payload is missing required aggregate fields.",
        "invalid_field_values": "Payload contains invalid field values.",
        "starter_not_supported_in_mvp": "Payload is not supported in MVP.",
        "internal_error": "Scoring failed.",
    }
)


@dataclass(frozen=True)
class ApiRuntime:
    """Immutable runtime loaded once at startup."""

    artifact_dir: str
    processed_dir: str
    processed_manifest: Mapping[str, Any]
    full_builder: Any
    reduced_builder: Any
    full_model: Any
    reduced_model: Any
    full_calibrator: Any
    reduced_calibrator: Any
    full_shap_explainer: Any
    reduced_shap_explainer: Any
    model_fairness_audit_passed: bool
    score_model_versions: Mapping[str, str]
    tier_metadata: Mapping[str, Mapping[str, Any]]
    health_model_version: str
    reproducibility_report: Mapping[str, Any]
    mock_mode: bool


@dataclass(frozen=True)
class ApiError(Exception):
    """Structured API error for stable JSON responses."""

    status_code: int
    error_code: str
    message: str
    missing_fields: tuple[str, ...] = ()

    def to_response(self):
        payload = {
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.status_code in (400, 422):
            payload["missing_fields"] = list(self.missing_fields)
        return jsonify(payload), self.status_code


class _MockBuilder:
    def __init__(self, tier: str):
        self.tier = tier.upper()
        base_columns = [
            "mock_income",
            "mock_credit",
            "mock_annuity",
            "mock_ratio",
            "mock_ext_mean",
            "mock_social",
            "mock_bureau",
            "mock_previous",
        ]
        if self.tier == "FULL":
            base_columns.extend(["mock_installments", "mock_pos", "mock_cc"])
        self.encoded_columns_ = base_columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        income = _numeric_series(df, "AMT_INCOME_TOTAL_CAPPED")
        credit = _numeric_series(df, "AMT_CREDIT")
        annuity = _numeric_series(df, "AMT_ANNUITY")
        ext1 = _numeric_series(df, "EXT_SOURCE_1")
        ext2 = _numeric_series(df, "EXT_SOURCE_2")
        ext3 = _numeric_series(df, "EXT_SOURCE_3")
        social = (
            _numeric_series(df, "OBS_30_CNT_SOCIAL_CIRCLE")
            + _numeric_series(df, "DEF_30_CNT_SOCIAL_CIRCLE")
            + _numeric_series(df, "OBS_60_CNT_SOCIAL_CIRCLE")
            + _numeric_series(df, "DEF_60_CNT_SOCIAL_CIRCLE")
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(income.to_numpy() > 0, credit.to_numpy() / income.to_numpy(), 0.0)

        out["mock_income"] = income.astype(float)
        out["mock_credit"] = credit.astype(float)
        out["mock_annuity"] = annuity.astype(float)
        out["mock_ratio"] = ratio.astype(float)
        out["mock_ext_mean"] = ((ext1 + ext2 + ext3) / 3.0).astype(float)
        out["mock_social"] = social.astype(float)
        out["mock_bureau"] = _numeric_series(df, "BUREAU_LOAN_COUNT")
        out["mock_previous"] = _numeric_series(df, "PREV_APP_COUNT")
        if self.tier == "FULL":
            out["mock_installments"] = _numeric_series(df, "INST_RECORD_COUNT")
            out["mock_pos"] = _numeric_series(df, "POS_RECORD_COUNT")
            out["mock_cc"] = _numeric_series(df, "CC_RECORD_COUNT")
        return out[self.encoded_columns_].astype(float)


class _MockModel:
    def __init__(self, feature_count: int):
        self.n_features_in_ = feature_count

    def predict_proba(self, X: Any) -> np.ndarray:
        arr = _coerce_2d_numeric(X)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features but received {arr.shape[1]}"
            )
        weights = np.linspace(0.15, 0.65, arr.shape[1], dtype=float)
        score = arr @ weights / max(arr.shape[1], 1)
        prob = 1.0 / (1.0 + np.exp(-(score / 100000.0)))
        prob = np.clip(prob, 0.01, 0.99)
        return np.column_stack([1.0 - prob, prob])


class _MockCalibrator:
    def predict(self, raw_pd: Any) -> np.ndarray:
        arr = np.asarray(raw_pd, dtype=float).reshape(-1)
        calibrated = 0.92 * arr + 0.03
        return np.clip(calibrated, 0.0, 1.0)


class _MockExplainer:
    def __call__(self, X: Any) -> np.ndarray:
        arr = _coerce_2d_numeric(X)
        weights = np.linspace(1.0, 2.0, arr.shape[1], dtype=float)
        return arr * weights


def create_app(
    artifact_dir: str | None = None,
    processed_dir: str | None = None,
    mock_mode: bool = False,
    strict_artifacts: bool = True,
) -> Flask:
    """Create the Flask app with eager startup validation."""

    resolved_artifact_dir, resolved_processed_dir = _resolve_runtime_dirs(
        artifact_dir,
        processed_dir,
        mock_mode=mock_mode,
    )

    runtime = (
        _build_mock_runtime(resolved_artifact_dir, resolved_processed_dir)
        if mock_mode
        else _load_real_runtime(
            artifact_dir=resolved_artifact_dir,
            processed_dir=resolved_processed_dir,
            strict_artifacts=strict_artifacts,
        )
    )

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.extensions[RUNTIME_EXTENSION_KEY] = runtime
    app.logger.setLevel(logging.INFO)

    @app.get("/")
    @app.get("/demo")
    def demo():
        loaded_runtime = _get_runtime(app)
        return render_template(
            "demo.html",
            demo_config=_build_demo_config(loaded_runtime),
        )

    @app.get("/health")
    def health():
        loaded_runtime = _get_runtime(app)
        return jsonify(
            {
                "status": "ok",
                "model_version": loaded_runtime.health_model_version,
                "fairness_audit_passed": loaded_runtime.model_fairness_audit_passed,
                "coverage_tiers_available": ["FULL", "REDUCED"],
            }
        )

    @app.post("/score")
    def score():
        request_id = _request_id()
        started_at = time.perf_counter()
        try:
            try:
                payload = request.get_json(silent=False)
            except Exception as exc:  # pragma: no cover - Flask wraps malformed JSON differently by version
                raise _api_error("bad_request") from exc

            if payload is None or not isinstance(payload, dict):
                raise _api_error("bad_request")

            loaded_runtime = _get_runtime(app)
            error_code, missing_fields = validate_payload(payload, loaded_runtime)
            if error_code is not None:
                raise _api_error(error_code, missing_fields)

            response = score_request(payload, loaded_runtime, mock_mode=mock_mode)
            _log_score_event(
                app.logger,
                endpoint="/score",
                request_id=request_id,
                payload=payload,
                response=response,
                started_at=started_at,
                status_code=200,
            )
            return jsonify(response)
        except ApiError as exc:
            _log_score_event(
                app.logger,
                endpoint="/score",
                request_id=request_id,
                payload=payload if "payload" in locals() else None,
                error_code=exc.error_code,
                started_at=started_at,
                status_code=exc.status_code,
            )
            return exc.to_response()
        except Exception:
            app.logger.exception("Scoring failed.")
            _log_score_event(
                app.logger,
                endpoint="/score",
                request_id=request_id,
                payload=payload if "payload" in locals() else None,
                error_code="internal_error",
                started_at=started_at,
                status_code=500,
            )
            return jsonify(
                {
                    "error_code": "internal_error",
                    "message": ERROR_MESSAGES["internal_error"],
                }
            ), 500

    @app.post("/score/batch")
    def score_batch():
        request_id = _request_id()
        started_at = time.perf_counter()
        try:
            try:
                payloads = request.get_json(silent=False)
            except Exception as exc:  # pragma: no cover - Flask wraps malformed JSON differently by version
                raise _api_error("bad_request") from exc

            if payloads is None or not isinstance(payloads, list):
                raise _api_error("bad_request")

            loaded_runtime = _get_runtime(app)
            results: list[dict[str, Any]] = []
            success_count = 0

            for index, payload in enumerate(payloads):
                if not isinstance(payload, dict):
                    results.append(
                        {
                            "index": index,
                            "ok": False,
                            "error": {
                                "error_code": "bad_request",
                                "message": ERROR_MESSAGES["bad_request"],
                                "missing_fields": [],
                            },
                        }
                    )
                    continue

                error_code, missing_fields = validate_payload(payload, loaded_runtime)
                if error_code is not None:
                    results.append(
                        {
                            "index": index,
                            "ok": False,
                            "error": {
                                "error_code": error_code,
                                "message": ERROR_MESSAGES[error_code],
                                "missing_fields": list(missing_fields),
                            },
                        }
                    )
                    continue

                try:
                    response = score_request(payload, loaded_runtime, mock_mode=mock_mode)
                    success_count += 1
                    results.append({"index": index, "ok": True, "response": response})
                except ApiError as exc:
                    results.append(
                        {
                            "index": index,
                            "ok": False,
                            "error": {
                                "error_code": exc.error_code,
                                "message": exc.message,
                                "missing_fields": list(exc.missing_fields),
                            },
                        }
                    )
                except Exception:
                    app.logger.exception("Batch scoring failed for index %s.", index)
                    results.append(
                        {
                            "index": index,
                            "ok": False,
                            "error": {
                                "error_code": "internal_error",
                                "message": ERROR_MESSAGES["internal_error"],
                                "missing_fields": [],
                            },
                        }
                    )

            batch_response = {
                "results": results,
                "summary": {
                    "total": len(results),
                    "succeeded": success_count,
                    "failed": len(results) - success_count,
                },
            }
            _log_batch_event(
                app.logger,
                request_id=request_id,
                payloads=payloads,
                batch_response=batch_response,
                started_at=started_at,
                status_code=200,
            )
            return jsonify(batch_response)
        except ApiError as exc:
            _log_batch_event(
                app.logger,
                request_id=request_id,
                payloads=payloads if "payloads" in locals() else None,
                batch_response=None,
                started_at=started_at,
                status_code=exc.status_code,
                error_code=exc.error_code,
            )
            return exc.to_response()
        except Exception:
            app.logger.exception("Batch scoring failed.")
            _log_batch_event(
                app.logger,
                request_id=request_id,
                payloads=payloads if "payloads" in locals() else None,
                batch_response=None,
                started_at=started_at,
                status_code=500,
                error_code="internal_error",
            )
            return jsonify(
                {
                    "error_code": "internal_error",
                    "message": ERROR_MESSAGES["internal_error"],
                }
            ), 500

    return app


def validate_payload(
    payload: dict,
    runtime: ApiRuntime | None = None,
) -> tuple[str | None, list[str]]:
    """Validate the public JSON contract and return an error code if invalid."""

    if not isinstance(payload, dict):
        return "bad_request", []

    unexpected_top_keys = sorted(set(payload) - ALLOWED_TOP_LEVEL_KEYS)
    if unexpected_top_keys:
        return "bad_request", []

    application = payload.get("application")
    if application is None:
        return "missing_application", ["application"]
    if not isinstance(application, dict):
        return "bad_request", []

    for section_name, section_value in payload.items():
        if not isinstance(section_value, dict):
            return "bad_request", []
        if not _section_has_scalar_values(section_value):
            return "bad_request", []

    if "CODE_GENDER" in application:
        return "forbidden_field_code_gender", []

    try:
        tier = determine_coverage_tier(payload)
    except ValueError:
        if set(payload).intersection(FULL_ONLY_SECTIONS):
            missing_sections = sorted(set(FULL_SECTION_ORDER) - set(payload))
            return "partial_full_payload_not_allowed", missing_sections
        return "starter_not_supported_in_mvp", []

    missing_application_fields = sorted(
        field for field in APPLICATION_REQUIRED_FIELDS if field not in application
    )
    if missing_application_fields:
        return "missing_application_fields", missing_application_fields

    if tier == "FULL":
        missing_aggregate_fields: list[str] = []
        for section_name in FULL_ONLY_SECTION_ORDER:
            section = payload.get(section_name)
            if section is None:
                missing_aggregate_fields.append(section_name)
                continue
            for field in AGG_REQUIRED_FIELDS[section_name]:
                if field not in section:
                    missing_aggregate_fields.append(f"{section_name}.{field}")
        if missing_aggregate_fields:
            return "missing_aggregate_fields", sorted(missing_aggregate_fields)

    invalid_fields = _collect_invalid_fields(payload, tier=tier, runtime=runtime)
    if invalid_fields:
        return "invalid_field_values", sorted(invalid_fields)

    return None, []


def _collect_invalid_fields(
    payload: dict,
    *,
    tier: str,
    runtime: ApiRuntime | None,
) -> list[str]:
    invalid_fields: list[str] = []
    application = payload.get("application", {})

    for field_name in NUMERIC_APPLICATION_FIELDS:
        if not _is_finite_number(application.get(field_name)):
            invalid_fields.append(field_name)

    if application.get("DAYS_EMPLOYED_ANOM") not in (0, 1, 0.0, 1.0):
        invalid_fields.append("DAYS_EMPLOYED_ANOM")

    for field_name in CATEGORICAL_APPLICATION_FIELDS:
        if not _is_valid_categorical_value(
            field_name,
            application.get(field_name),
            tier=tier,
            runtime=runtime,
        ):
            invalid_fields.append(field_name)

    if tier == "FULL":
        for section_name in FULL_ONLY_SECTION_ORDER:
            section = payload.get(section_name, {})
            for field_name in NUMERIC_FIELDS_BY_SECTION[section_name]:
                if not _is_finite_number(section.get(field_name)):
                    invalid_fields.append(f"{section_name}.{field_name}")

    return invalid_fields


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(float(value)))
    return False


def _is_valid_categorical_value(
    field_name: str,
    value: Any,
    *,
    tier: str,
    runtime: ApiRuntime | None,
) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    allowed_values = _allowed_categorical_values(field_name, tier=tier, runtime=runtime)
    if allowed_values and value not in allowed_values:
        return False
    return True


def _allowed_categorical_values(
    field_name: str,
    *,
    tier: str,
    runtime: ApiRuntime | None,
) -> set[str]:
    if runtime is not None and not runtime.mock_mode:
        builder = runtime.full_builder if tier == "FULL" else runtime.reduced_builder
        rare_map = getattr(builder, "rare_category_maps_", {})
        fill_values = getattr(builder, "categorical_fill_values_", {})
        allowed = set(str(v) for v in rare_map.get(field_name, set()))
        fill_value = fill_values.get(field_name)
        if fill_value:
            allowed.add(str(fill_value))
        if allowed:
            allowed.add("OTHER")
        return allowed
    return set(DEMO_FIELD_OPTIONS.get(field_name, ()))


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _request_id() -> str:
    header_value = request.headers.get("X-Request-ID")
    return header_value or uuid.uuid4().hex


def _sanitize_payload_summary(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        return {"batch_size": len(payload)}
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}
    summary: dict[str, Any] = {"top_level_keys": sorted(payload.keys())}
    application = payload.get("application")
    if isinstance(application, dict):
        summary["application_field_count"] = len(application)
    for section_name in FULL_ONLY_SECTION_ORDER:
        section = payload.get(section_name)
        if isinstance(section, dict):
            summary[f"{section_name}_field_count"] = len(section)
    return summary


def _log_score_event(
    logger: logging.Logger,
    *,
    endpoint: str,
    request_id: str,
    payload: Any,
    started_at: float,
    status_code: int,
    response: Mapping[str, Any] | None = None,
    error_code: str | None = None,
) -> None:
    event = {
        "timestamp": _utc_now_iso(),
        "request_id": request_id,
        "endpoint": endpoint,
        "status_code": status_code,
        "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
        "payload_summary": _sanitize_payload_summary(payload),
        "error_code": error_code,
    }
    if response is not None:
        event.update(
            {
                "tier": response.get("coverage_tier"),
                "probability_of_default": response.get("probability_of_default"),
                "decision": response.get("decision"),
                "model_version": response.get("model_version"),
            }
        )
    logger.info(json.dumps(event, default=str))


def _log_batch_event(
    logger: logging.Logger,
    *,
    request_id: str,
    payloads: Any,
    batch_response: Mapping[str, Any] | None,
    started_at: float,
    status_code: int,
    error_code: str | None = None,
) -> None:
    summary = _sanitize_payload_summary(payloads)
    event = {
        "timestamp": _utc_now_iso(),
        "request_id": request_id,
        "endpoint": "/score/batch",
        "status_code": status_code,
        "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
        "payload_summary": summary,
        "error_code": error_code,
    }
    if batch_response is not None:
        event["batch_summary"] = dict(batch_response.get("summary", {}))
    logger.info(json.dumps(event, default=str))


def determine_coverage_tier(payload: dict) -> str:
    """Classify payloads as FULL or REDUCED based on top-level sections."""

    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if "application" not in payload or payload.get("application") is None:
        raise ValueError("application is required")

    top_keys = set(payload)
    if top_keys == {"application"}:
        return "REDUCED"
    if top_keys == set(FULL_SECTION_ORDER):
        return "FULL"
    if top_keys.intersection(FULL_ONLY_SECTIONS):
        raise ValueError("partial full payload")
    raise ValueError("unsupported starter payload")


def build_input_df(payload: dict, tier: str) -> pd.DataFrame:
    """Flatten a valid request payload into a one-row DataFrame."""

    tier = tier.upper()
    if tier not in {"FULL", "REDUCED"}:
        raise ValueError("tier must be FULL or REDUCED")

    flattened: dict[str, Any] = {}
    section_names = ("application",) if tier == "REDUCED" else FULL_SECTION_ORDER

    for section_name in section_names:
        section = payload.get(section_name)
        if section_name == "application" and section is None:
            raise _api_error("missing_application", ["application"])
        if not isinstance(section, dict):
            raise _api_error("bad_request")

        for field_name, field_value in section.items():
            if not _is_scalar_value(field_value):
                raise _api_error("bad_request")
            if field_name in flattened:
                raise _api_error("bad_request")
            flattened[field_name] = field_value

    return pd.DataFrame([flattened])


def score_request(payload: dict, runtime: ApiRuntime, mock_mode: bool = False) -> dict:
    """Score one request using the loaded immutable runtime."""

    error_code, missing_fields = validate_payload(payload, runtime)
    if error_code is not None:
        raise _api_error(error_code, missing_fields)

    tier = determine_coverage_tier(payload)
    input_df = build_input_df(payload, tier)
    builder = runtime.full_builder if tier == "FULL" else runtime.reduced_builder
    model = runtime.full_model if tier == "FULL" else runtime.reduced_model
    calibrator = runtime.full_calibrator if tier == "FULL" else runtime.reduced_calibrator
    explainer = runtime.full_shap_explainer if tier == "FULL" else runtime.reduced_shap_explainer
    requires_linear_features = bool(runtime.tier_metadata[tier]["requires_linear_features"])

    features = _invoke_builder(builder, input_df, for_linear_model=requires_linear_features)
    raw_pd = _predict_raw_pd(model, features)
    calibrated_pd = _calibrate_pd(calibrator, raw_pd)
    decision = trained_decision_from_pd(calibrated_pd)
    explanations = (
        _mock_top_5_explanations(features.columns)
        if mock_mode
        else _compute_real_top_5_explanations(explainer, features)
    )

    return {
        "probability_of_default": calibrated_pd,
        "decision": decision,
        "escalate": decision == "REVIEW",
        "top_5_explanations": explanations,
        "model_version": runtime.score_model_versions[tier],
        "calibrated": True,
        "model_fairness_audit_passed": runtime.model_fairness_audit_passed,
        "fairness_audit_version": FAIRNESS_AUDIT_VERSION,
        "coverage_tier": tier,
    }


def _resolve_runtime_dirs(
    artifact_dir: str | None,
    processed_dir: str | None,
    mock_mode: bool,
) -> tuple[str, str]:
    if mock_mode:
        resolved_artifact_dir = artifact_dir or tempfile.mkdtemp(prefix="mastermind_mock_artifacts_")
        resolved_processed_dir = processed_dir or tempfile.mkdtemp(prefix="mastermind_mock_processed_")
    else:
        resolved_artifact_dir = artifact_dir or ARTIFACT_DIR
        resolved_processed_dir = processed_dir or DATA_DIR
    return (os.path.abspath(resolved_artifact_dir), os.path.abspath(resolved_processed_dir))


def _build_mock_runtime(artifact_dir: str, processed_dir: str) -> ApiRuntime:
    full_builder = _MockBuilder("FULL")
    reduced_builder = _MockBuilder("REDUCED")
    full_model = _MockModel(len(full_builder.encoded_columns_))
    reduced_model = _MockModel(len(reduced_builder.encoded_columns_))
    full_calibrator = _MockCalibrator()
    reduced_calibrator = _MockCalibrator()
    full_explainer = _MockExplainer()
    reduced_explainer = _MockExplainer()

    processed_manifest = MappingProxyType(
        {
            "mode": "mock",
            "processed_manifest_id": "mock-processed-manifest",
        }
    )
    score_versions = MappingProxyType(
        {
            "FULL": _build_composite_model_version("FULL"),
            "REDUCED": _build_composite_model_version("REDUCED"),
        }
    )
    tier_metadata = MappingProxyType(
        {
            "FULL": MappingProxyType(
                {
                    "model_family": "mock",
                    "requires_linear_features": False,
                    "version": score_versions["FULL"],
                }
            ),
            "REDUCED": MappingProxyType(
                {
                    "model_family": "mock",
                    "requires_linear_features": False,
                    "version": score_versions["REDUCED"],
                }
            ),
        }
    )

    return ApiRuntime(
        artifact_dir=artifact_dir,
        processed_dir=processed_dir,
        processed_manifest=processed_manifest,
        full_builder=full_builder,
        reduced_builder=reduced_builder,
        full_model=full_model,
        reduced_model=reduced_model,
        full_calibrator=full_calibrator,
        reduced_calibrator=reduced_calibrator,
        full_shap_explainer=full_explainer,
        reduced_shap_explainer=reduced_explainer,
        model_fairness_audit_passed=False,
        score_model_versions=score_versions,
        tier_metadata=tier_metadata,
        health_model_version=score_versions["FULL"],
        reproducibility_report=MappingProxyType({"mode": "mock"}),
        mock_mode=True,
    )


def _load_real_runtime(
    artifact_dir: str,
    processed_dir: str,
    strict_artifacts: bool,
) -> ApiRuntime:
    processed_manifest = _load_processed_manifest(processed_dir)
    builder_module = _import_builder_artifacts_module()
    full_builder, reduced_builder = _load_validated_builders(
        builder_module,
        artifact_dir=artifact_dir,
        processed_dir=processed_dir,
        processed_manifest=processed_manifest,
        strict_artifacts=strict_artifacts,
    )
    artifacts = load_artifacts(artifact_dir)
    reproducibility_report = artifacts["reproducibility_report"]
    tier_metadata = _resolve_tier_metadata(reproducibility_report)
    processed_manifest_id = processed_manifest.get("processed_manifest_id")
    for tier_name in ("FULL", "REDUCED"):
        if tier_metadata[tier_name].get("processed_manifest_id") != processed_manifest_id:
            raise RuntimeError(f"{tier_name} artifact manifest mismatch")
    full_model = artifacts["full_model"]
    reduced_model = artifacts["reduced_model"]
    full_calibrator = artifacts["full_calibrator"]
    reduced_calibrator = artifacts["reduced_calibrator"]
    full_explainer = artifacts["full_shap_explainer"]
    reduced_explainer = artifacts["reduced_shap_explainer"]
    fairness_result = _load_fairness_result(artifact_dir)

    _validate_tier_runtime(
        "FULL",
        full_builder,
        full_model,
        full_calibrator,
        full_explainer,
        requires_linear_features=bool(tier_metadata["FULL"]["requires_linear_features"]),
    )
    _validate_tier_runtime(
        "REDUCED",
        reduced_builder,
        reduced_model,
        reduced_calibrator,
        reduced_explainer,
        requires_linear_features=bool(tier_metadata["REDUCED"]["requires_linear_features"]),
    )

    score_versions = MappingProxyType(_resolve_score_versions(reproducibility_report))
    health_model_version = _resolve_health_model_version(
        reproducibility_report,
        fallback=score_versions["FULL"],
    )

    return ApiRuntime(
        artifact_dir=artifact_dir,
        processed_dir=processed_dir,
        processed_manifest=MappingProxyType(processed_manifest),
        full_builder=full_builder,
        reduced_builder=reduced_builder,
        full_model=full_model,
        reduced_model=reduced_model,
        full_calibrator=full_calibrator,
        reduced_calibrator=reduced_calibrator,
        full_shap_explainer=full_explainer,
        reduced_shap_explainer=reduced_explainer,
        model_fairness_audit_passed=fairness_result,
        score_model_versions=score_versions,
        tier_metadata=tier_metadata,
        health_model_version=health_model_version,
        reproducibility_report=MappingProxyType(reproducibility_report),
        mock_mode=False,
    )


def _load_processed_manifest(processed_dir: str) -> dict[str, Any]:
    manifest_path = os.path.join(processed_dir, PROCESSED_MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"Missing required processed manifest: {manifest_path}")
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to read processed manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("processed_artifact_manifest.json must contain a JSON object")
    return manifest


def _import_builder_artifacts_module() -> ModuleType:
    try:
        return importlib.import_module("src.builder_artifacts")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "src.builder_artifacts.py is required for strict builder loading"
        ) from exc


def _load_validated_builders(
    builder_module: ModuleType,
    *,
    artifact_dir: str,
    processed_dir: str,
    processed_manifest: Mapping[str, Any],
    strict_artifacts: bool,
) -> tuple[Any, Any]:
    common_kwargs = {
        "artifact_dir": artifact_dir,
        "processed_dir": processed_dir,
        "processed_manifest": processed_manifest,
        "strict_artifacts": strict_artifacts,
    }

    batch_loader_names = (
        "load_validated_builders",
        "load_builder_runtime",
        "load_builders",
    )
    for loader_name in batch_loader_names:
        loader = getattr(builder_module, loader_name, None)
        if callable(loader):
            result = _call_with_supported_kwargs(loader, **common_kwargs)
            return _coerce_builder_pair(result)

    generic_loader = getattr(builder_module, "load_builder", None)
    if callable(generic_loader):
        full_builder = _call_with_supported_kwargs(generic_loader, tier="FULL", **common_kwargs)
        reduced_builder = _call_with_supported_kwargs(generic_loader, tier="REDUCED", **common_kwargs)
        return full_builder, reduced_builder

    full_loader = getattr(builder_module, "load_full_builder", None)
    reduced_loader = getattr(builder_module, "load_reduced_builder", None)
    if callable(full_loader) and callable(reduced_loader):
        full_builder = _call_with_supported_kwargs(full_loader, tier="FULL", **common_kwargs)
        reduced_builder = _call_with_supported_kwargs(reduced_loader, tier="REDUCED", **common_kwargs)
        return full_builder, reduced_builder

    raise RuntimeError(
        "src.builder_artifacts.py does not expose a supported builder loader/validator interface"
    )


def _coerce_builder_pair(result: Any) -> tuple[Any, Any]:
    if isinstance(result, dict):
        full_builder = result.get("full_builder") or result.get("FULL") or result.get("full")
        reduced_builder = (
            result.get("reduced_builder") or result.get("REDUCED") or result.get("reduced")
        )
        if full_builder is not None and reduced_builder is not None:
            return full_builder, reduced_builder

    if isinstance(result, (tuple, list)) and len(result) == 2:
        return result[0], result[1]

    full_builder = getattr(result, "full_builder", None)
    reduced_builder = getattr(result, "reduced_builder", None)
    if full_builder is not None and reduced_builder is not None:
        return full_builder, reduced_builder

    raise RuntimeError("Builder loader must return FULL and REDUCED builders together")


def _call_with_supported_kwargs(func: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(func)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return func(**kwargs)
    supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return func(**supported_kwargs)


def _load_joblib_artifact(artifact_dir: str, filename: str, label: str) -> Any:
    path = os.path.join(artifact_dir, filename)
    if not os.path.exists(path):
        raise RuntimeError(f"Missing required {label}: {path}")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {label}: {path}") from exc


def _load_optional_json(artifact_dir: str, filename: str) -> dict[str, Any]:
    path = os.path.join(artifact_dir, filename)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON metadata: {path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON metadata must be an object: {path}")
    return data


def _load_fairness_result(artifact_dir: str) -> bool:
    fairness_obj = _load_joblib_artifact(
        artifact_dir,
        FAIRNESS_RESULT_FILENAME,
        "fairness result artifact",
    )
    if isinstance(fairness_obj, (bool, np.bool_)):
        return bool(fairness_obj)
    raise RuntimeError(
        "model_fairness_audit_passed.joblib must contain a boolean result"
    )


def _validate_tier_runtime(
    tier: str,
    builder: Any,
    model: Any,
    calibrator: Any,
    explainer: Any,
    *,
    requires_linear_features: bool = False,
) -> None:
    smoke_payload = _build_smoke_payload(tier)
    smoke_df = build_input_df(smoke_payload, tier)
    features = _invoke_builder(builder, smoke_df, for_linear_model=requires_linear_features)

    if features.shape[0] != 1:
        raise RuntimeError(f"{tier} builder smoke transform must return exactly one row")

    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and int(n_features) != features.shape[1]:
        raise RuntimeError(
            f"{tier} model expects {int(n_features)} features but builder produced {features.shape[1]}"
        )

    raw_pd = _predict_raw_pd(model, features)
    calibrated_pd = _calibrate_pd(calibrator, raw_pd)
    if not 0.0 <= calibrated_pd <= 1.0:
        raise RuntimeError(f"{tier} calibrator produced an invalid probability")

    _compute_real_top_5_explanations(explainer, features)


def _build_smoke_payload(tier: str) -> dict[str, Any]:
    demo_payload = _build_demo_seed_payload()
    if tier.upper() == "REDUCED":
        return {"application": dict(demo_payload["application"])}
    return demo_payload


def _build_demo_seed_payload() -> dict[str, Any]:
    application = {
        "AMT_INCOME_TOTAL_CAPPED": 120000.0,
        "AMT_CREDIT": 250000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 220000.0,
        "DAYS_BIRTH": -12000.0,
        "DAYS_EMPLOYED": -1500.0,
        "DAYS_REGISTRATION": -3000.0,
        "DAYS_ID_PUBLISH": -2000.0,
        "DAYS_LAST_PHONE_CHANGE": -1000.0,
        "REGION_POPULATION_RELATIVE": 0.02,
        "EXT_SOURCE_1": 0.2,
        "EXT_SOURCE_2": 0.4,
        "EXT_SOURCE_3": 0.6,
        "CNT_FAM_MEMBERS": 2.0,
        "OWN_CAR_AGE": 5.0,
        "OBS_30_CNT_SOCIAL_CIRCLE": 1.0,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
        "OBS_60_CNT_SOCIAL_CIRCLE": 1.0,
        "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
        "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
        "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
        "AMT_REQ_CREDIT_BUREAU_WEEK": 1.0,
        "AMT_REQ_CREDIT_BUREAU_MON": 1.0,
        "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "OCCUPATION_TYPE": "Laborers",
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
        "DAYS_EMPLOYED_ANOM": 0,
    }
    return {
        "application": application,
        "bureau_agg": {
            "BUREAU_LOAN_COUNT": 2.0,
            "BUREAU_ACTIVE_COUNT": 1.0,
            "BUREAU_CLOSED_COUNT": 1.0,
            "BUREAU_AMT_CREDIT_SUM_SUM": 50000.0,
            "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": 10000.0,
            "BUREAU_DEBT_TO_CREDIT_RATIO": 0.2,
            "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM": 0.0,
            "BUREAU_CREDIT_DAY_OVERDUE_MAX": 0.0,
            "BUREAU_DAYS_CREDIT_MAX": -200.0,
            "BUREAU_CNT_CREDIT_PROLONG_SUM": 0.0,
        },
        "previous_agg": {
            "PREV_APP_COUNT": 3.0,
            "PREV_APPROVED_COUNT": 2.0,
            "PREV_REFUSED_COUNT": 1.0,
            "PREV_APPROVAL_RATE": 0.67,
            "PREV_REFUSAL_RATE": 0.33,
            "PREV_AMT_APPLICATION_MEAN": 210000.0,
            "PREV_AMT_CREDIT_MEAN": 195000.0,
            "PREV_AMT_GOODS_PRICE_MEAN": 187000.0,
            "PREV_APP_CREDIT_DIFF_MEAN": 15000.0,
            "PREV_DAYS_DECISION_MAX": -120.0,
            "PREV_RATE_DOWN_PAYMENT_MEAN": 0.08,
        },
        "installments_agg": {
            "INST_RECORD_COUNT": 12.0,
            "INST_MISSED_RATE": 0.08,
            "INST_DPD_MEAN": 4.0,
            "INST_DPD_MAX": 12.0,
            "INST_PAYMENT_RATIO_MEAN": 0.95,
            "INST_PAYMENT_RATIO_MIN": 0.72,
            "INST_LATE_COUNT": 3.0,
        },
        "pos_cash_agg": {
            "POS_RECORD_COUNT": 6.0,
            "POS_DPD_MEAN": 1.5,
            "POS_DPD_MAX": 7.0,
            "POS_DPD_DEF_MEAN": 0.5,
            "POS_DPD_DEF_MAX": 4.0,
            "POS_COMPLETED_RATE": 0.6,
            "POS_ACTIVE_RATE": 0.4,
            "POS_CNT_INSTALMENT_FUTURE_MEAN": 2.0,
        },
        "credit_card_agg": {
            "CC_RECORD_COUNT": 8.0,
            "CC_BALANCE_MEAN": 18000.0,
            "CC_LIMIT_MEAN": 60000.0,
            "CC_UTILIZATION_MEAN": 0.3,
            "CC_PAYMENT_RATIO_MEAN": 1.1,
            "CC_DPD_MEAN": 1.0,
            "CC_DPD_MAX": 6.0,
            "CC_DRAWINGS_ATM_SUM": 4500.0,
            "CC_DRAWINGS_CURRENT_SUM": 9000.0,
        },
    }


def _build_demo_config(runtime: ApiRuntime) -> dict[str, Any]:
    sample_payload = _build_demo_seed_payload()
    sections: list[dict[str, Any]] = []

    for section_name in FULL_SECTION_ORDER:
        field_names = (
            APPLICATION_REQUIRED_FIELDS
            if section_name == "application"
            else AGG_REQUIRED_FIELDS[section_name]
        )
        fields = []
        for field_name in field_names:
            field_kind = "select" if field_name in CATEGORICAL_APPLICATION_FIELDS else "number"
            fields.append(
                {
                    "name": field_name,
                    "kind": field_kind,
                    "options": list(DEMO_FIELD_OPTIONS.get(field_name, ())),
                }
            )
        sections.append(
            {
                "name": section_name,
                "label": section_name.replace("_", " ").title(),
                "fields": fields,
            }
        )

    return {
        "runtimeMode": "mock" if runtime.mock_mode else "real",
        "healthModelVersion": runtime.health_model_version,
        "scoreModelVersions": dict(runtime.score_model_versions),
        "fairnessAuditPassed": runtime.model_fairness_audit_passed,
        "fairnessAuditVersion": FAIRNESS_AUDIT_VERSION,
        "defaultTier": DEMO_DEFAULT_TIER,
        "sections": sections,
        "samplePayloads": {
            "REDUCED": {"application": dict(sample_payload["application"])},
            "FULL": sample_payload,
        },
        "routes": {
            "health": "/health",
            "score": "/score",
            "scoreBatch": "/score/batch",
        },
    }


def _invoke_builder(
    builder: Any,
    df: pd.DataFrame,
    *,
    for_linear_model: bool = False,
) -> pd.DataFrame:
    if hasattr(builder, "transform") and callable(builder.transform):
        result = _call_with_supported_kwargs(
            builder.transform,
            df=df,
            for_linear_model=for_linear_model,
        )
    elif callable(builder):
        result = builder(df)
    else:
        raise RuntimeError("Loaded builder is neither callable nor transformable")

    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("Builder output must be a pandas DataFrame")
    if result.shape[0] != df.shape[0]:
        raise RuntimeError("Builder output row count does not match input row count")
    return result


def _predict_raw_pd(model: Any, features: pd.DataFrame) -> float:
    if not hasattr(model, "predict_proba") or not callable(model.predict_proba):
        raise RuntimeError("Loaded model does not expose predict_proba")
    probs = np.asarray(model.predict_proba(features), dtype=float)
    if probs.ndim != 2 or probs.shape[0] != 1 or probs.shape[1] < 2:
        raise RuntimeError("Model predict_proba must return shape (1, >=2)")
    return float(probs[0, 1])


def _calibrate_pd(calibrator: Any, raw_pd: float) -> float:
    if not hasattr(calibrator, "predict") or not callable(calibrator.predict):
        raise RuntimeError("Loaded calibrator does not expose predict")
    calibrated = np.asarray(calibrator.predict(np.array([raw_pd], dtype=float)), dtype=float).reshape(-1)
    if calibrated.size != 1:
        raise RuntimeError("Calibrator predict must return exactly one probability")
    return float(np.clip(calibrated[0], 0.0, 1.0))


def _decision_from_pd(probability_of_default: float) -> str:
    return trained_decision_from_pd(probability_of_default)


def _compute_real_top_5_explanations(explainer: Any, features: pd.DataFrame) -> list[dict[str, str]]:
    shap_series = _compute_shap_series(explainer, features)
    explanations = top_5_explanations_from_shap(shap_series)
    return _ensure_five_explanations(explanations, list(shap_series.index))


def _compute_shap_series(explainer: Any, features: pd.DataFrame) -> pd.Series:
    raw_input = features.to_numpy(dtype=float, copy=False)
    if callable(explainer):
        raw_shap = explainer(raw_input)
    elif hasattr(explainer, "shap_values") and callable(explainer.shap_values):
        raw_shap = explainer.shap_values(raw_input)
    else:
        raise RuntimeError("Loaded explainer is not callable and has no shap_values method")
    values = _normalize_shap_output(raw_shap, len(features.columns))
    return pd.Series(values, index=features.columns, dtype=float)


def _normalize_shap_output(raw_shap: Any, feature_count: int) -> np.ndarray:
    try:
        import shap  # type: ignore
    except Exception:  # pragma: no cover - dependency import differences are environment-specific
        shap = None

    if shap is not None and isinstance(raw_shap, shap.Explanation):
        raw_shap = raw_shap.values

    if isinstance(raw_shap, list):
        if not raw_shap:
            raise RuntimeError("Explainer returned an empty SHAP list")
        raw_shap = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]

    values = np.asarray(raw_shap, dtype=float)
    if values.ndim == 3:
        class_index = 1 if values.shape[-1] > 1 else 0
        values = values[..., class_index]
    if values.ndim == 2:
        if values.shape[0] == 1:
            values = values[0]
        elif values.shape[1] == 1:
            values = values[:, 0]
    if values.ndim != 1 or values.shape[0] != feature_count:
        raise RuntimeError("Explainer returned SHAP values with an unexpected shape")
    return values


def _ensure_five_explanations(
    explanations: list[dict[str, str]],
    feature_names: list[str],
) -> list[dict[str, str]]:
    if len(explanations) >= 5:
        return explanations[:5]
    used = {item["feature"] for item in explanations}
    for feature_name in feature_names:
        if feature_name in used:
            continue
        explanations.append({"feature": feature_name, "reason": render_reason(feature_name)})
        used.add(feature_name)
        if len(explanations) == 5:
            break
    while len(explanations) < 5:
        fallback_feature = f"fallback_feature_{len(explanations) + 1}"
        explanations.append({"feature": fallback_feature, "reason": render_reason(fallback_feature)})
    return explanations


def _mock_top_5_explanations(feature_names: Any) -> list[dict[str, str]]:
    ordered_names = list(feature_names)[:5]
    if len(ordered_names) < 5:
        ordered_names.extend(f"mock_feature_{index}" for index in range(len(ordered_names), 5))
    return [{"feature": name, "reason": render_reason(name)} for name in ordered_names[:5]]


def _coerce_2d_numeric(value: Any) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        arr = value.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _numeric_series(df: pd.DataFrame, column_name: str, default: float = 0.0) -> pd.Series:
    if column_name in df.columns:
        return pd.to_numeric(df[column_name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _section_has_scalar_values(section: Mapping[str, Any]) -> bool:
    return all(_is_scalar_value(value) for value in section.values())


def _is_scalar_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool, np.generic)):
        return True
    return False


def _resolve_tier_metadata(reproducibility_report: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    tiers = reproducibility_report.get("tiers", {})
    if not isinstance(tiers, dict):
        raise RuntimeError("reproducibility_report.json must contain a tiers object")
    out: dict[str, Mapping[str, Any]] = {}
    for tier_name in ("FULL", "REDUCED"):
        tier_meta = tiers.get(tier_name)
        if not isinstance(tier_meta, dict):
            raise RuntimeError(f"reproducibility_report.json missing {tier_name} tier metadata")
        out[tier_name] = MappingProxyType(dict(tier_meta))
    return MappingProxyType(out)


def _resolve_score_versions(reproducibility_report: Mapping[str, Any]) -> dict[str, str]:
    report_versions = reproducibility_report.get("model_versions")
    if isinstance(report_versions, dict):
        full_base = report_versions.get("FULL")
        reduced_base = report_versions.get("REDUCED")
        if isinstance(full_base, str) and isinstance(reduced_base, str):
            return {
                "FULL": _build_composite_model_version("FULL", base_version=full_base),
                "REDUCED": _build_composite_model_version("REDUCED", base_version=reduced_base),
            }
    return {
        "FULL": _build_composite_model_version("FULL"),
        "REDUCED": _build_composite_model_version("REDUCED"),
    }


def _build_composite_model_version(tier: str, base_version: str | None = None) -> str:
    tier_key = tier.lower()
    tier_version = base_version or MODEL_VERSIONS.get(tier_key, f"{tier_key}_unknown")
    fairness_tag = _fairness_tag_from_version(FAIRNESS_AUDIT_VERSION)
    return f"{tier_version}|{ROUTER_VERSION}|{POLICY_VERSION}|fairness_v{fairness_tag}"


def _fairness_tag_from_version(version: str) -> str:
    match = re.search(r"(20\d{2}Q[1-4])", version)
    return match.group(1) if match else DEFAULT_FAIRNESS_VERSION_TAG


def _resolve_health_model_version(
    reproducibility_report: Mapping[str, Any],
    *,
    fallback: str,
) -> str:
    preferred_keys = (
        "deployed_model_version",
        "health_model_version",
        "model_version",
        "champion_model_version",
        "full_model_version",
    )
    discovered = _find_first_string_value(reproducibility_report, preferred_keys)
    return discovered or fallback


def _find_first_string_value(data: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        for value in data.values():
            found = _find_first_string_value(value, keys)
            if found:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _find_first_string_value(value, keys)
            if found:
                return found
    return None


def _get_runtime(app: Flask) -> ApiRuntime:
    runtime = app.extensions.get(RUNTIME_EXTENSION_KEY)
    if runtime is None:
        raise RuntimeError("API runtime is not initialized")
    return runtime


def _api_error(error_code: str, missing_fields: list[str] | tuple[str, ...] | None = None) -> ApiError:
    status_code = 400 if error_code == "bad_request" else 422
    return ApiError(
        status_code=status_code,
        error_code=error_code,
        message=ERROR_MESSAGES[error_code],
        missing_fields=tuple(missing_fields or ()),
    )


__all__ = [
    "APPLICATION_REQUIRED_FIELDS",
    "AGG_REQUIRED_FIELDS",
    "ApiRuntime",
    "build_input_df",
    "create_app",
    "determine_coverage_tier",
    "score_request",
    "validate_payload",
]
