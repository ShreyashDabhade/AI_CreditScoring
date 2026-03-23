from __future__ import annotations

from dataclasses import is_dataclass, replace
import json
from pathlib import Path
import sys
from types import ModuleType

import joblib
import numpy as np
import pandas as pd

from src.api import (
    AGG_REQUIRED_FIELDS,
    APPLICATION_REQUIRED_FIELDS,
    build_input_df,
    create_app,
    determine_coverage_tier,
    validate_payload,
)


def _make_application_payload() -> dict[str, object]:
    values: dict[str, object] = {
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
    assert set(APPLICATION_REQUIRED_FIELDS).issubset(values)
    return values


def _make_full_payload() -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {"application": _make_application_payload()}
    for section_name, fields in AGG_REQUIRED_FIELDS.items():
        payload[section_name] = {
            field: float(index + 1) for index, field in enumerate(fields)
        }
    return payload


class SavedBuilderFixture:
    def __init__(self, tier: str):
        self.tier = tier.upper()
        self.columns = [
            f"{self.tier.lower()}_feature_1",
            f"{self.tier.lower()}_feature_2",
            f"{self.tier.lower()}_feature_3",
            f"{self.tier.lower()}_feature_4",
            f"{self.tier.lower()}_feature_5",
            f"{self.tier.lower()}_feature_6",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        base = pd.DataFrame(index=df.index)
        base[self.columns[0]] = pd.to_numeric(df.get("AMT_INCOME_TOTAL_CAPPED", 0.0), errors="coerce").fillna(0.0)
        base[self.columns[1]] = pd.to_numeric(df.get("AMT_CREDIT", 0.0), errors="coerce").fillna(0.0)
        base[self.columns[2]] = pd.to_numeric(df.get("AMT_ANNUITY", 0.0), errors="coerce").fillna(0.0)
        base[self.columns[3]] = pd.to_numeric(df.get("EXT_SOURCE_1", 0.0), errors="coerce").fillna(0.0)
        base[self.columns[4]] = pd.to_numeric(df.get("EXT_SOURCE_2", 0.0), errors="coerce").fillna(0.0)
        if self.tier == "FULL":
            base[self.columns[5]] = pd.to_numeric(df.get("BUREAU_LOAN_COUNT", 0.0), errors="coerce").fillna(0.0)
        else:
            base[self.columns[5]] = pd.to_numeric(df.get("DAYS_EMPLOYED_ANOM", 0.0), errors="coerce").fillna(0.0)
        return base[self.columns].astype(float)


class SavedModelFixture:
    def __init__(self, feature_count: int):
        self.n_features_in_ = feature_count

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        score = arr.mean(axis=1) / 100000.0
        prob = 1.0 / (1.0 + np.exp(-score))
        prob = np.clip(prob, 0.01, 0.99)
        return np.column_stack([1.0 - prob, prob])


class SavedCalibratorFixture:
    def predict(self, raw_pd):
        arr = np.asarray(raw_pd, dtype=float).reshape(-1)
        return np.clip(0.95 * arr + 0.01, 0.0, 1.0)


class SavedExplainerFixture:
    def __call__(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        weights = np.linspace(1.0, 2.0, arr.shape[1], dtype=float)
        return arr * weights


class ExplodingModel:
    def predict_proba(self, X):
        raise RuntimeError("boom")


def _install_fake_builder_module(monkeypatch, loader) -> None:
    module = ModuleType("src.builder_artifacts")
    module.load_validated_builders = loader
    monkeypatch.setitem(sys.modules, "src.builder_artifacts", module)


def _write_processed_manifest(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "processed_manifest_id": "processed-001",
        "lineage": "processed-001",
    }
    (processed_dir / "processed_artifact_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _write_real_artifacts(artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    full_builder = SavedBuilderFixture("FULL")
    joblib.dump(SavedModelFixture(len(full_builder.columns)), artifact_dir / "full_model.joblib")
    joblib.dump(SavedCalibratorFixture(), artifact_dir / "full_calibrator.joblib")
    joblib.dump(SavedExplainerFixture(), artifact_dir / "full_shap_explainer.joblib")
    joblib.dump(True, artifact_dir / "model_fairness_audit_passed.joblib")
    (artifact_dir / "reproducibility_report.json").write_text(
        json.dumps({"deployed_model_version": "deployed-full-2026.03"}),
        encoding="utf-8",
    )


def test_create_app_mock_mode_starts_successfully():
    app = create_app(mock_mode=True)
    runtime = app.extensions["mastermind_runtime"]

    assert is_dataclass(runtime)
    assert runtime.mock_mode is True
    assert Path(runtime.artifact_dir).exists()
    assert Path(runtime.processed_dir).exists()


def test_mock_mode_default_does_not_use_shared_artifact_dir():
    project_root = Path(__file__).resolve().parents[1]
    shared_artifact_dir = project_root / "artifacts"
    before = sorted(path.name for path in shared_artifact_dir.iterdir()) if shared_artifact_dir.exists() else []

    app = create_app(mock_mode=True)
    runtime = app.extensions["mastermind_runtime"]

    after = sorted(path.name for path in shared_artifact_dir.iterdir()) if shared_artifact_dir.exists() else []
    assert Path(runtime.artifact_dir).resolve() != shared_artifact_dir.resolve()
    assert before == after


def test_health_returns_required_fields():
    client = create_app(mock_mode=True).test_client()

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert set(payload) == {
        "status",
        "model_version",
        "fairness_audit_passed",
        "coverage_tiers_available",
    }
    assert payload["status"] == "ok"
    assert payload["coverage_tiers_available"] == ["FULL"]


def test_demo_routes_render_frontend():
    client = create_app(mock_mode=True).test_client()

    root_response = client.get("/")
    demo_response = client.get("/demo")

    assert root_response.status_code == 200
    assert demo_response.status_code == 200
    assert "MasterMind Demo Console" in root_response.get_data(as_text=True)
    assert "window.__MASTERMIND_DEMO__" in demo_response.get_data(as_text=True)


def test_score_application_only_returns_422():
    client = create_app(mock_mode=True).test_client()

    response = client.post("/score", json={"application": _make_application_payload()})

    assert response.status_code == 422
    payload = response.get_json()
    assert payload["error_code"] == "starter_not_supported_in_mvp"


def test_score_valid_full_returns_200():
    client = create_app(mock_mode=True).test_client()

    response = client.post("/score", json=_make_full_payload())

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["coverage_tier"] == "FULL"
    assert payload["model_version"].startswith("full_v")


def test_bad_json_returns_400():
    client = create_app(mock_mode=True).test_client()

    response = client.post("/score", data="{", content_type="application/json")

    assert response.status_code == 400
    assert response.get_json()["error_code"] == "bad_request"


def test_missing_application_returns_422():
    client = create_app(mock_mode=True).test_client()

    response = client.post("/score", json={"bureau_agg": {}})

    assert response.status_code == 422
    payload = response.get_json()
    assert payload["error_code"] == "missing_application"
    assert payload["missing_fields"] == ["application"]


def test_code_gender_returns_422():
    client = create_app(mock_mode=True).test_client()
    payload = {"application": _make_application_payload()}
    payload["application"]["CODE_GENDER"] = "M"

    response = client.post("/score", json=payload)

    assert response.status_code == 422
    assert response.get_json()["error_code"] == "forbidden_field_code_gender"


def test_missing_application_fields_returns_422():
    client = create_app(mock_mode=True).test_client()
    payload = {"application": _make_application_payload()}
    payload["application"].pop("AMT_CREDIT")

    response = client.post("/score", json=payload)

    assert response.status_code == 422
    body = response.get_json()
    assert body["error_code"] == "missing_application_fields"
    assert "AMT_CREDIT" in body["missing_fields"]


def test_partial_full_payload_returns_422():
    client = create_app(mock_mode=True).test_client()
    payload = {"application": _make_application_payload(), "bureau_agg": {}}

    response = client.post("/score", json=payload)

    assert response.status_code == 422
    assert response.get_json()["error_code"] == "partial_full_payload_not_allowed"


def test_missing_aggregate_fields_returns_422():
    client = create_app(mock_mode=True).test_client()
    payload = _make_full_payload()
    payload["bureau_agg"].pop("BUREAU_LOAN_COUNT")

    response = client.post("/score", json=payload)

    assert response.status_code == 422
    body = response.get_json()
    assert body["error_code"] == "missing_aggregate_fields"
    assert "bureau_agg.BUREAU_LOAN_COUNT" in body["missing_fields"]


def test_validate_payload_rejects_non_object_sections_and_non_scalar_values():
    bad_app_code, _ = validate_payload({"application": []})
    bad_nested_code, _ = validate_payload({"application": {"AMT_CREDIT": [1, 2, 3]}})
    bad_full_code, _ = validate_payload(
        {"application": _make_application_payload(), "bureau_agg": []}
    )

    assert bad_app_code == "bad_request"
    assert bad_nested_code == "bad_request"
    assert bad_full_code == "bad_request"


def test_validate_payload_and_determine_coverage_tier_happy_paths():
    full = _make_full_payload()

    assert determine_coverage_tier(full) == "FULL"
    assert validate_payload(full) == (None, [])
    assert validate_payload({"application": _make_application_payload()}) == (
        "starter_not_supported_in_mvp",
        [],
    )


def test_validate_payload_rejects_unknown_top_level_keys():
    error_code, missing_fields = validate_payload(
        {"application": _make_application_payload(), "unknown_section": {}}
    )

    assert error_code == "bad_request"
    assert missing_fields == []


def test_build_input_df_builds_full_payload():
    full_df = build_input_df(_make_full_payload(), "FULL")

    assert full_df.shape[0] == 1
    assert "AMT_CREDIT" in full_df.columns
    assert "BUREAU_LOAN_COUNT" in full_df.columns
    assert "CC_RECORD_COUNT" in full_df.columns


def test_build_input_df_rejects_non_full_tier():
    try:
        build_input_df(_make_full_payload(), "REDUCED")
        assert False, "Expected non-FULL tier rejection"
    except ValueError as exc:
        assert "FULL" in str(exc)


def test_build_input_df_rejects_duplicate_flattened_columns():
    payload = _make_full_payload()
    payload["bureau_agg"]["AMT_CREDIT"] = 99.0

    try:
        build_input_df(payload, "FULL")
        assert False, "Expected duplicate flattened field rejection"
    except Exception as exc:
        assert "bad_request" in str(exc) or exc.__class__.__name__ == "ApiError"


def test_strict_real_mode_fails_when_processed_manifest_is_missing(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"
    artifact_dir.mkdir()
    processed_dir.mkdir()

    try:
        create_app(
            artifact_dir=str(artifact_dir),
            processed_dir=str(processed_dir),
            mock_mode=False,
        )
        assert False, "Expected missing processed manifest startup failure"
    except RuntimeError as exc:
        assert "processed manifest" in str(exc)


def test_strict_real_mode_fails_on_builder_lineage_mismatch(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"
    _write_processed_manifest(processed_dir)

    def loader(**kwargs):
        raise RuntimeError("builder lineage mismatch")

    _install_fake_builder_module(monkeypatch, loader)

    try:
        create_app(
            artifact_dir=str(artifact_dir),
            processed_dir=str(processed_dir),
            mock_mode=False,
        )
        assert False, "Expected builder lineage mismatch startup failure"
    except RuntimeError as exc:
        assert "lineage mismatch" in str(exc)


def test_strict_real_mode_fails_when_non_builder_artifact_is_missing(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"
    _write_processed_manifest(processed_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def loader(**kwargs):
        return {
            "full_builder": SavedBuilderFixture("FULL"),
        }

    _install_fake_builder_module(monkeypatch, loader)

    try:
        create_app(
            artifact_dir=str(artifact_dir),
            processed_dir=str(processed_dir),
            mock_mode=False,
        )
        assert False, "Expected missing non-builder artifact startup failure"
    except RuntimeError as exc:
        assert "FULL model" in str(exc)


def test_real_mode_health_prefers_reproducibility_report_metadata(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"
    _write_processed_manifest(processed_dir)
    _write_real_artifacts(artifact_dir)

    def loader(**kwargs):
        return {
            "full_builder": SavedBuilderFixture("FULL"),
        }

    _install_fake_builder_module(monkeypatch, loader)
    app = create_app(
        artifact_dir=str(artifact_dir),
        processed_dir=str(processed_dir),
        mock_mode=False,
    )

    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json()["model_version"] == "deployed-full-2026.03"


def test_internal_error_path_returns_500():
    app = create_app(mock_mode=True)
    runtime = app.extensions["mastermind_runtime"]
    app.extensions["mastermind_runtime"] = replace(runtime, full_model=ExplodingModel())
    client = app.test_client()

    response = client.post("/score", json=_make_full_payload())

    assert response.status_code == 500
    assert response.get_json() == {
        "error_code": "internal_error",
        "message": "Scoring failed.",
    }
