from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.api.minimal_app import SECTION_FIELD_MAP, create_app
from src.db_manager import create_application, update_application


class ConstantModel:
    def __init__(self, feature_count: int, probability: float):
        self.feature_count = feature_count
        self.probability = probability

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=float)
        assert arr.shape[1] == self.feature_count
        prob = np.full(arr.shape[0], self.probability, dtype=float)
        return np.column_stack([1.0 - prob, prob])


class IdentityCalibrator:
    def predict(self, raw_pd):
        return np.asarray(raw_pd, dtype=float)


def _build_application_payload() -> dict[str, object]:
    return {
        "AMT_INCOME_TOTAL": 250000.0,
        "AMT_CREDIT": 500000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "DAYS_BIRTH": -14000.0,
        "DAYS_EMPLOYED": -2200.0,
        "DAYS_REGISTRATION": -3200.0,
        "DAYS_ID_PUBLISH": -1800.0,
        "DAYS_LAST_PHONE_CHANGE": -250.0,
        "REGION_POPULATION_RELATIVE": 0.018,
        "EXT_SOURCE_1": 0.72,
        "EXT_SOURCE_2": 0.64,
        "EXT_SOURCE_3": 0.58,
        "CNT_FAM_MEMBERS": 3.0,
        "OWN_CAR_AGE": 4.0,
        "OBS_30_CNT_SOCIAL_CIRCLE": 1.0,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
        "OBS_60_CNT_SOCIAL_CIRCLE": 1.0,
        "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
        "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
        "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
        "AMT_REQ_CREDIT_BUREAU_WEEK": 1.0,
        "AMT_REQ_CREDIT_BUREAU_MON": 1.0,
        "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 2.0,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "OCCUPATION_TYPE": "Laborers",
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
    }


def _build_full_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "tier": "FULL",
        "application": _build_application_payload(),
    }
    for section_name, fields in SECTION_FIELD_MAP.items():
        payload[section_name] = {
            field: float(index + 1) for index, field in enumerate(fields)
        }
    return payload


def _write_runtime_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact_dir = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"
    db_path = tmp_path / "applications.sqlite3"
    artifact_dir.mkdir()
    processed_dir.mkdir()

    full_builder = joblib.load("artifacts/full_feature_builder.joblib")
    reduced_builder = joblib.load("artifacts/reduced_feature_builder.joblib")

    joblib.dump(full_builder, artifact_dir / "full_feature_builder.joblib")
    joblib.dump(reduced_builder, artifact_dir / "reduced_feature_builder.joblib")
    joblib.dump(
        ConstantModel(len(full_builder.encoded_columns_), 0.12),
        artifact_dir / "full_model.joblib",
    )
    joblib.dump(
        ConstantModel(len(reduced_builder.encoded_columns_), 0.21),
        artifact_dir / "reduced_model.joblib",
    )
    joblib.dump(IdentityCalibrator(), artifact_dir / "full_calibrator.joblib")
    joblib.dump(IdentityCalibrator(), artifact_dir / "reduced_calibrator.joblib")
    joblib.dump(True, artifact_dir / "model_fairness_audit_passed.joblib")
    joblib.dump(300000.0, processed_dir / "income_cap.joblib")
    return artifact_dir, processed_dir, db_path


def test_health_reports_runtime_state(tmp_path):
    artifact_dir, processed_dir, db_path = _write_runtime_artifacts(tmp_path)
    app = create_app(str(artifact_dir), str(processed_dir), str(db_path))
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model_version"] == "full_v2.1.0"
    assert body["coverage_tiers_available"] == ["FULL", "REDUCED"]
    assert body["fairness_audit_passed"] is True


def test_score_reduced_runs_saved_builder_and_artifacts(tmp_path):
    artifact_dir, processed_dir, db_path = _write_runtime_artifacts(tmp_path)
    app = create_app(str(artifact_dir), str(processed_dir), str(db_path))
    client = app.test_client()

    response = client.post(
        "/score",
        json={"tier": "REDUCED", "application": _build_application_payload()},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["coverage_tier"] == "REDUCED"
    assert body["decision"] == "REVIEW"
    assert body["calibrated"] is True
    assert body["probability_of_default"] == 0.21
    assert body["escalate"] is True
    assert body["model_fairness_audit_passed"] is True
    assert body["fairness_audit_version"] == "proxy_audit_2026Q1_v1.1"
    assert len(body["top_5_explanations"]) == 5


def test_score_full_runs_saved_builder_and_artifacts(tmp_path):
    artifact_dir, processed_dir, db_path = _write_runtime_artifacts(tmp_path)
    app = create_app(str(artifact_dir), str(processed_dir), str(db_path))
    client = app.test_client()

    response = client.post("/score", json=_build_full_payload())

    assert response.status_code == 200
    body = response.get_json()
    assert body["coverage_tier"] == "FULL"
    assert body["decision"] == "APPROVE"
    assert body["probability_of_default"] == 0.12
    assert body["escalate"] is False
    assert len(body["top_5_explanations"]) == 5


def test_chat_health_and_chat_route_use_existing_chatbot_logic(tmp_path, monkeypatch):
    artifact_dir, processed_dir, db_path = _write_runtime_artifacts(tmp_path)

    monkeypatch.setattr("src.api.minimal_app.is_gemini_available", lambda: True)
    monkeypatch.setattr(
        "src.api.minimal_app.chatbot_chat",
        lambda **kwargs: {
            "response": f"echo:{kwargs['message']}",
            "source": "fallback",
            "gemini_available": False,
        },
    )

    app = create_app(str(artifact_dir), str(processed_dir), str(db_path))
    client = app.test_client()

    health_response = client.get("/api/chat/health")
    assert health_response.status_code == 200
    assert health_response.get_json() == {"status": "ok", "gemini_available": True}

    chat_response = client.post(
        "/api/chat",
        json={
            "message": "hello",
            "context": {"score_data": {"decision": "Review"}},
            "history": [{"role": "user", "content": "prev"}],
        },
    )
    assert chat_response.status_code == 200
    assert chat_response.get_json() == {
        "response": "echo:hello",
        "source": "fallback",
        "gemini_available": False,
    }


def test_simulate_calls_score_pipeline_and_returns_frontend_shape(tmp_path):
    artifact_dir, processed_dir, db_path = _write_runtime_artifacts(tmp_path)
    app = create_app(str(artifact_dir), str(processed_dir), str(db_path))
    client = app.test_client()

    application = create_application(
        str(db_path),
        applicant_name="Test Applicant",
        tier_type="REDUCED",
        application_payload_json={"application": _build_application_payload()},
        current_status="ANALYZED",
        sk_id_curr=123456,
    )
    update_application(
        str(db_path),
        int(application["id"]),
        last_probability=0.30,
        last_decision="DECLINE",
        last_model_version="reduced_v2.1.0",
    )

    response = client.post(
        f"/analyst/applications/{application['id']}/simulate",
        json={"changes": {"AMT_CREDIT": "350000"}},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["application_id"] == application["id"]
    assert body["coverage_tier"] == "REDUCED"
    assert body["original_probability"] == 0.30
    assert body["original_probability_text"] == "30.0%"
    assert body["original_decision"] == "DECLINE"
    assert body["original_decision_label"] == "Decline"
    assert body["simulated_probability"] == 0.21
    assert body["simulated_probability_text"] == "21.0%"
    assert body["simulated_decision"] == "REVIEW"
    assert body["simulated_decision_label"] == "Review"
    assert body["simulated_decision_badge_class"] == "review"
    assert body["delta"] == -0.09
    assert body["delta_text"] == "-9.0 pts"
    assert body["delta_direction"] == "down"
    assert body["decision_delta_label"] == "Decline -> Review"
    assert body["decision_changed"] is True
    assert body["risk_movement_label"] == "Risk decreased"
    assert body["change_count"] == 1
    assert body["model_version"] == "reduced_v2.1.0"
    assert body["fairness_audit_passed"] is True
    assert body["non_persistent"] is True
    assert len(body["top_5_explanations"]) == 5
    assert body["changed_features"] == [
        {
            "label": "Amt Credit",
            "field": "application.AMT_CREDIT",
            "before": "500,000",
            "after": "350,000",
        }
    ]


def test_ui_pages_render_with_existing_templates(tmp_path):
    artifact_dir, processed_dir, db_path = _write_runtime_artifacts(tmp_path)
    app = create_app(str(artifact_dir), str(processed_dir), str(db_path))
    client = app.test_client()

    for path in ("/", "/analyze", "/status", "/analytics", "/applications/new", "/analyst"):
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.content_type


def test_report_page_renders_simulator_contract(tmp_path):
    artifact_dir, processed_dir, db_path = _write_runtime_artifacts(tmp_path)
    app = create_app(str(artifact_dir), str(processed_dir), str(db_path))
    client = app.test_client()

    application = create_application(
        str(db_path),
        applicant_name="Report Applicant",
        tier_type="REDUCED",
        application_payload_json={"application": _build_application_payload()},
        current_status="SUBMITTED",
        sk_id_curr=654321,
    )

    analyze_response = client.post(f"/analyst/applications/{application['id']}/analyze")
    assert analyze_response.status_code == 302
    assert analyze_response.headers["Location"].endswith(f"/analyst/applications/{application['id']}/report")

    report_response = client.get(f"/analyst/applications/{application['id']}/report")
    assert report_response.status_code == 200
    html = report_response.get_data(as_text=True)
    assert "window.__REPORT_SIMULATOR__" in html
    assert f"/analyst/applications/{application['id']}/simulate" in html
    assert "window.__REPORT_COPILOT__" in html
