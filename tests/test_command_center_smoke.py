from __future__ import annotations

import importlib
import json
from pathlib import Path

from src.api.app import _build_demo_seed_payload
from src.db_manager import get_application_by_id, init_database


def _load_ui_app(tmp_path: Path):
    ui_module = importlib.import_module("app")
    db_path = init_database(tmp_path / "smoke.sqlite3")
    ui_module.app.config["APPLICATION_DB_PATH"] = db_path
    ui_module.app.testing = True
    return ui_module.app, db_path


def test_database_initialization_smoke(tmp_path: Path):
    db_path = Path(init_database(tmp_path / "command_center.sqlite3"))

    assert db_path.exists()


def test_application_lifecycle_smoke(tmp_path: Path):
    app, db_path = _load_ui_app(tmp_path)
    payload = {"application": _build_demo_seed_payload()["application"]}
    client = app.test_client()

    created_response = client.post(
        "/applications",
        data={
            "applicant_name": "Smoke Applicant",
            "tier_type": "REDUCED",
            "current_status": "SUBMITTED",
            "application_payload_json": json.dumps(payload),
        },
        follow_redirects=False,
    )
    assert created_response.status_code == 302

    application = get_application_by_id(db_path, 1)
    assert application is not None

    updated_response = client.post(
        "/analyst/applications/1/update",
        data={
            "applicant_name": "Smoke Applicant Updated",
            "current_status": "READY_FOR_REVIEW",
            "application_payload_json": json.dumps(payload),
        },
        follow_redirects=False,
    )
    assert updated_response.status_code == 302

    analyzed_response = client.post(
        "/analyst/applications/1/analyze",
        follow_redirects=False,
    )
    assert analyzed_response.status_code == 302

    report_response = client.get("/analyst/applications/1/report")
    assert report_response.status_code == 200
    assert "Deep Insight Report" in report_response.get_data(as_text=True)


def test_existing_routes_and_score_smoke(tmp_path: Path):
    app, _ = _load_ui_app(tmp_path)
    client = app.test_client()
    payload = {"application": _build_demo_seed_payload()["application"]}

    assert client.get("/").status_code == 200
    assert client.get("/analyze").status_code == 200
    assert client.get("/status").status_code == 200
    assert client.get("/analytics").status_code == 200

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.get_json()["status"] == "ok"

    score_response = client.post("/score", json=payload)
    assert score_response.status_code == 200
    score_json = score_response.get_json()
    assert score_json["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
