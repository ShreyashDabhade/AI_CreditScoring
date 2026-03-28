from __future__ import annotations

import importlib
import json
from pathlib import Path

from src.api.app import _build_demo_seed_payload
from src.db_manager import (
    create_application,
    get_application_by_id,
    init_database,
    list_score_history_for_application,
)


def _load_ui_app(tmp_path: Path):
    ui_module = importlib.import_module("app")
    db_path = init_database(tmp_path / "command_center.sqlite3")
    ui_module.app.config["APPLICATION_DB_PATH"] = db_path
    ui_module.app.testing = True
    return ui_module.app, db_path


def test_get_applications_new_page_renders(tmp_path: Path):
    app, _ = _load_ui_app(tmp_path)

    response = app.test_client().get("/applications/new")

    assert response.status_code == 200
    assert "Create a saved application" in response.get_data(as_text=True)


def test_post_applications_creates_record_and_redirects(tmp_path: Path):
    app, db_path = _load_ui_app(tmp_path)
    reduced_payload = {"application": _build_demo_seed_payload()["application"]}

    response = app.test_client().post(
        "/applications",
        data={
            "applicant_name": "Casey Applicant",
            "tier_type": "REDUCED",
            "current_status": "SUBMITTED",
            "application_payload_json": json.dumps(reduced_payload),
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    location = response.headers["Location"]
    assert "/applications/" in location

    created = get_application_by_id(db_path, 1)
    assert created is not None
    assert created["applicant_name"] == "Casey Applicant"
    assert created["current_status"] == "SUBMITTED"


def test_analyst_routes_render_and_return_json(tmp_path: Path):
    app, db_path = _load_ui_app(tmp_path)
    reduced_payload = {"application": _build_demo_seed_payload()["application"]}
    create_application(
        db_path,
        applicant_name="Taylor Analyst",
        tier_type="REDUCED",
        current_status="READY_FOR_REVIEW",
        application_payload_json=reduced_payload,
    )

    client = app.test_client()
    dashboard = client.get("/analyst")
    listing = client.get("/analyst/applications?format=json")
    detail = client.get("/analyst/applications/1")

    assert dashboard.status_code == 200
    assert "Risk operations dashboard" in dashboard.get_data(as_text=True)
    assert listing.status_code == 200
    assert listing.get_json()["applications"][0]["applicant_name"] == "Taylor Analyst"
    assert detail.status_code == 200
    assert "Application #1" in detail.get_data(as_text=True)


def test_analyst_update_and_analyze_flow(tmp_path: Path):
    app, db_path = _load_ui_app(tmp_path)
    reduced_payload = {"application": _build_demo_seed_payload()["application"]}
    application = create_application(
        db_path,
        applicant_name="Jordan Review",
        tier_type="REDUCED",
        current_status="READY_FOR_REVIEW",
        application_payload_json=reduced_payload,
    )

    client = app.test_client()
    update_response = client.post(
        f"/analyst/applications/{application['id']}/update",
        data={
            "applicant_name": "Jordan Reviewer",
            "current_status": "READY_FOR_REVIEW",
            "application_payload_json": json.dumps(reduced_payload),
        },
        follow_redirects=False,
    )
    analyze_response = client.post(
        f"/analyst/applications/{application['id']}/analyze",
        follow_redirects=False,
    )

    assert update_response.status_code == 302
    assert analyze_response.status_code == 302
    assert analyze_response.headers["Location"].endswith(
        f"/analyst/applications/{application['id']}/report"
    )

    refreshed = get_application_by_id(db_path, application["id"])
    history = list_score_history_for_application(db_path, application["id"])

    assert refreshed is not None
    assert refreshed["applicant_name"] == "Jordan Reviewer"
    assert refreshed["current_status"] in {"APPROVED", "REVIEW", "DECLINED"}
    assert refreshed["last_decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    assert len(history) == 1

    report = client.get(f"/analyst/applications/{application['id']}/report")
    assert report.status_code == 200
    report_html = report.get_data(as_text=True)
    assert "Deep Insight Report" in report_html
    assert "Risk posture" in report_html
    assert "What-If Simulator" in report_html


def test_reanalyze_retains_prior_score_history(tmp_path: Path):
    app, db_path = _load_ui_app(tmp_path)
    reduced_payload = {"application": _build_demo_seed_payload()["application"]}
    application = create_application(
        db_path,
        applicant_name="Repeat Runner",
        tier_type="REDUCED",
        current_status="READY_FOR_REVIEW",
        application_payload_json=reduced_payload,
    )

    client = app.test_client()
    first = client.post(f"/analyst/applications/{application['id']}/analyze", follow_redirects=False)
    second = client.post(f"/analyst/applications/{application['id']}/analyze", follow_redirects=False)

    assert first.status_code == 302
    assert second.status_code == 302

    history = list_score_history_for_application(db_path, application["id"])
    refreshed = get_application_by_id(db_path, application["id"])

    assert len(history) == 2
    assert refreshed is not None
    assert refreshed["last_decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    assert refreshed["current_status"] in {"APPROVED", "REVIEW", "DECLINED"}


def test_report_simulator_returns_temporary_result_without_persistence(tmp_path: Path):
    app, db_path = _load_ui_app(tmp_path)
    reduced_payload = {"application": _build_demo_seed_payload()["application"]}
    application = create_application(
        db_path,
        applicant_name="Scenario Analyst",
        tier_type="REDUCED",
        current_status="READY_FOR_REVIEW",
        application_payload_json=reduced_payload,
    )

    client = app.test_client()
    analyze_response = client.post(
        f"/analyst/applications/{application['id']}/analyze",
        follow_redirects=False,
    )
    assert analyze_response.status_code == 302

    before = get_application_by_id(db_path, application["id"])
    before_history = list_score_history_for_application(db_path, application["id"])

    simulate_response = client.post(
        f"/analyst/applications/{application['id']}/simulate",
        json={
            "changes": {
                "AMT_CREDIT": 310000,
                "EXT_SOURCE_2": 0.55,
            }
        },
    )

    assert simulate_response.status_code == 200
    payload = simulate_response.get_json()
    assert payload["non_persistent"] is True
    assert payload["change_count"] == 2
    assert payload["simulated_decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    assert payload["simulated_probability"] is not None

    after = get_application_by_id(db_path, application["id"])
    after_history = list_score_history_for_application(db_path, application["id"])

    assert before is not None
    assert after is not None
    assert before["last_probability"] == after["last_probability"]
    assert before["last_decision"] == after["last_decision"]
    assert len(before_history) == len(after_history) == 1
