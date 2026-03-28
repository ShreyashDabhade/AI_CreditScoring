from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.api import create_app
from src.db_manager import (
    create_application,
    get_application_by_id,
    init_database,
    list_application_change_history,
    list_applications,
    list_score_history_for_application,
    save_application_change_log,
    save_score_run,
    update_application,
)


def test_init_database_creates_expected_tables(tmp_path: Path):
    db_path = Path(init_database(tmp_path / "loan_apps.sqlite3"))

    assert db_path.exists()

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()

    table_names = {row[0] for row in rows}
    assert "application_change_log" in table_names
    assert "loan_applications" in table_names
    assert "score_runs" in table_names


def test_application_crud_round_trip(tmp_path: Path):
    db_path = init_database(tmp_path / "loan_apps.sqlite3")
    created = create_application(
        db_path,
        applicant_name="Jane Doe",
        tier_type="reduced",
        current_status="submitted",
        application_payload_json={"application": {"AMT_CREDIT": 250000.0}},
        sk_id_curr=100001,
    )

    assert created["applicant_name"] == "Jane Doe"
    assert created["tier_type"] == "REDUCED"
    assert created["current_status"] == "SUBMITTED"
    assert created["sk_id_curr"] == 100001
    assert json.loads(created["application_payload_json"]) == {
        "application": {"AMT_CREDIT": 250000.0}
    }

    listed = list_applications(db_path)
    assert [row["id"] for row in listed] == [created["id"]]

    fetched = get_application_by_id(db_path, created["id"])
    assert fetched is not None
    assert fetched["id"] == created["id"]

    updated = update_application(
        db_path,
        created["id"],
        applicant_name="Jane R. Doe",
        current_status="ready_for_review",
        application_payload_json={"application": {"AMT_CREDIT": 275000.0}},
    )

    assert updated is not None
    assert updated["applicant_name"] == "Jane R. Doe"
    assert updated["current_status"] == "READY_FOR_REVIEW"
    assert json.loads(updated["application_payload_json"]) == {
        "application": {"AMT_CREDIT": 275000.0}
    }


def test_save_score_run_updates_application_and_lists_history(tmp_path: Path):
    db_path = init_database(tmp_path / "loan_apps.sqlite3")
    application = create_application(
        db_path,
        applicant_name="John Smith",
        tier_type="FULL",
        application_payload_json={"application": {"AMT_CREDIT": 100000.0}},
        current_status="SUBMITTED",
    )

    score_run = save_score_run(
        db_path,
        application_id=application["id"],
        probability=0.37,
        decision="REVIEW",
        model_version="reduced_v2.1.0",
        score_payload_json={
            "probability_of_default": 0.37,
            "decision": "REVIEW",
            "model_version": "reduced_v2.1.0",
        },
    )

    assert score_run["application_id"] == application["id"]
    assert score_run["probability"] == 0.37
    assert score_run["decision"] == "REVIEW"

    refreshed = get_application_by_id(db_path, application["id"])
    assert refreshed is not None
    assert refreshed["current_status"] == "REVIEW"
    assert refreshed["last_probability"] == 0.37
    assert refreshed["last_decision"] == "REVIEW"
    assert refreshed["last_model_version"] == "reduced_v2.1.0"

    history = list_score_history_for_application(db_path, application["id"])
    assert len(history) == 1
    assert history[0]["id"] == score_run["id"]
    assert json.loads(history[0]["score_payload_json"]) == {
        "decision": "REVIEW",
        "model_version": "reduced_v2.1.0",
        "probability_of_default": 0.37,
    }


def test_create_app_initializes_application_database(tmp_path: Path):
    db_path = tmp_path / "startup.sqlite3"

    app = create_app(mock_mode=True, application_db_path=str(db_path))

    assert app.config["APPLICATION_DB_PATH"] == str(db_path.resolve())
    assert db_path.exists()

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('loan_applications', 'score_runs', 'application_change_log')"
        ).fetchall()

    assert {row[0] for row in rows} == {"loan_applications", "score_runs", "application_change_log"}


def test_save_application_change_log_and_list_history(tmp_path: Path):
    db_path = init_database(tmp_path / "loan_apps.sqlite3")
    application = create_application(
        db_path,
        applicant_name="Audit Trail",
        tier_type="REDUCED",
        application_payload_json={"application": {"AMT_CREDIT": 150000.0}},
        current_status="SUBMITTED",
    )

    change_log = save_application_change_log(
        db_path,
        application_id=application["id"],
        change_summary_json={
            "application_id": application["id"],
            "change_count": 2,
            "changes": [
                {"field": "applicant_name", "before": "Audit Trail", "after": "Audit Trail 2"},
                {"field": "application.AMT_CREDIT", "before": 150000.0, "after": 175000.0},
            ],
        },
    )

    history = list_application_change_history(db_path, application["id"])

    assert change_log["application_id"] == application["id"]
    assert len(history) == 1
    payload = json.loads(history[0]["change_summary_json"])
    assert payload["change_count"] == 2
    assert payload["changes"][0]["field"] == "applicant_name"
