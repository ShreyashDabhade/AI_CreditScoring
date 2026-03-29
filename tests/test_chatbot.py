from __future__ import annotations

import sqlite3

import src.chatbot as chatbot
from src import query_parser
from src.api.agent_routes import run_pipeline
from src.db_manager import create_application, init_database, save_score_run


def _seed_application(
    db_path: str,
    *,
    application_id: int,
    sk_id_curr: int,
    applicant_name: str,
    probability: float,
    decision: str,
    income: float,
    credit: float,
    ext1: float,
    ext2: float,
    ext3: float,
) -> None:
    create_application(
        db_path,
        applicant_name=applicant_name,
        tier_type="REDUCED",
        current_status="READY_FOR_REVIEW",
        sk_id_curr=sk_id_curr,
        application_payload_json={
            "application": {
                "AMT_INCOME_TOTAL_CAPPED": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": credit / 10,
                "DAYS_BIRTH": -365 * 32,
                "DAYS_EMPLOYED": -365 * 4,
                "NAME_EDUCATION_TYPE": "Higher education",
                "EXT_SOURCE_1": ext1,
                "EXT_SOURCE_2": ext2,
                "EXT_SOURCE_3": ext3,
            }
        },
    )
    save_score_run(
        db_path,
        application_id=application_id,
        probability=probability,
        decision=decision,
        model_version="reduced_v2.1.0",
        score_payload_json={
            "probability_of_default": probability,
            "decision": decision,
            "coverage_tier": "REDUCED",
            "model_version": "reduced_v2.1.0",
            "top_5_explanations": [
                {"feature": "Credit Amount", "reason": "Requested credit remains high relative to income."},
                {"feature": "External Source 2", "reason": "External credit reference signals were weaker than preferred."},
            ],
        },
    )


def _report_context() -> dict:
    return {
        "scope": "analyst_application_report",
        "application_id": 7,
        "applicant_summary": {
            "applicant_name": "Grounded Analyst Case",
            "coverage_tier": "REDUCED",
            "current_status": "REVIEW",
        },
        "latest_assessment": {
            "decision": "REVIEW",
            "decision_label": "Review",
            "calibrated_probability": 0.184,
            "calibrated_probability_text": "18.4%",
            "decision_summary": "The calibrated risk signal requires analyst review before final action.",
            "model_version": "reduced_v2.1.0",
        },
        "top_drivers": [
            {
                "feature": "External Source 2",
                "reason": "External reference signals were weaker than preferred.",
                "rank": 1,
            },
            {
                "feature": "Credit Amount",
                "reason": "Requested credit remains elevated relative to current income.",
                "rank": 2,
            },
        ],
        "shap_narratives": [
            "External Source 2 is the leading risk driver in this assessment because external reference signals were weaker than preferred."
        ],
        "score_history": [
            {
                "scored_at": "29 Mar 2026, 10:45 UTC",
                "decision": "REVIEW",
                "decision_label": "Review",
                "probability_text": "18.4%",
                "model_version": "reduced_v2.1.0",
                "is_latest": True,
            }
        ],
        "adverse_action": {
            "section_title": "Adverse Action Summary",
            "summary": "External reference quality and affordability signals kept this case in review.",
            "reasons": [
                {
                    "title": "External credit reference strength",
                    "detail": "External credit reference signals were weaker than preferred for this policy.",
                },
                {
                    "title": "Loan affordability pressure",
                    "detail": "Requested credit appears high relative to current reported income.",
                },
            ],
        },
        "simulator_result": {
            "original_probability_text": "18.4%",
            "original_decision_label": "Review",
            "simulated_probability_text": "14.9%",
            "simulated_decision_label": "Approve",
            "delta_text": "-3.5 pts",
            "risk_movement_label": "Risk decreased",
            "changed_features": [
                {
                    "label": "Credit Amount",
                    "before": "300,000",
                    "after": "250,000",
                }
            ],
        },
    }


def test_chat_uses_grounded_report_context_for_risk_summary():
    result = chatbot.chat("Summarize the risk profile", report_context=_report_context())

    assert result["source"] == "fallback"
    assert "18.4%" in result["response"]
    assert "External Source 2" in result["response"]
    assert "Grounded Analyst Case" in result["response"]


def test_chat_uses_grounded_report_context_for_simulator_questions():
    result = chatbot.chat("What changes reduced risk in the simulator?", report_context=_report_context())

    assert result["source"] == "fallback"
    assert "14.9%" in result["response"]
    assert "Risk decreased" in result["response"]
    assert "Credit Amount" in result["response"]


def test_chat_uses_pipeline_backend_for_applicant_explanations(monkeypatch, tmp_path):
    db_path = init_database(tmp_path / "chatbot.sqlite3")
    create_application(
        db_path,
        applicant_name="Pipeline Case",
        tier_type="REDUCED",
        current_status="READY_FOR_REVIEW",
        sk_id_curr=12345,
        application_payload_json={
            "application": {
                "AMT_INCOME_TOTAL_CAPPED": 80000.0,
                "AMT_CREDIT": 240000.0,
                "AMT_ANNUITY": 24000.0,
                "DAYS_BIRTH": -365 * 32,
                "DAYS_EMPLOYED": -365 * 3,
                "NAME_EDUCATION_TYPE": "Higher education",
                "EXT_SOURCE_1": 0.42,
                "EXT_SOURCE_2": 0.46,
                "EXT_SOURCE_3": 0.4,
            }
        },
    )
    save_score_run(
        db_path,
        application_id=1,
        probability=0.41,
        decision="DECLINE",
        model_version="reduced_v2.1.0",
        score_payload_json={
            "probability_of_default": 0.41,
            "decision": "DECLINE",
            "coverage_tier": "REDUCED",
            "model_version": "reduced_v2.1.0",
            "top_5_explanations": [
                {"feature": "Credit Amount", "reason": "Requested credit remains high relative to income."},
                {"feature": "External Source 2", "reason": "External credit reference signals were weaker than preferred."},
            ],
        },
    )
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))

    result = chatbot.chat("Why was applicant 12345 rejected?")

    assert result["source"] == "fallback"
    assert "Applicant 12345" in result["response"]
    assert "declined" in result["response"].lower()
    assert "41.0%" in result["response"]
    assert "risk" in result["response"].lower()


def test_run_pipeline_accepts_lookup_query_against_command_center_db(monkeypatch, tmp_path):
    db_path = init_database(tmp_path / "lookup.sqlite3")
    _seed_application(
        db_path,
        application_id=1,
        sk_id_curr=54321,
        applicant_name="Lookup Case",
        probability=0.12,
        decision="APPROVE",
        income=120000.0,
        credit=200000.0,
        ext1=0.7,
        ext2=0.75,
        ext3=0.72,
    )
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))

    result = run_pipeline("show applicant 54321")

    assert result["row_count"] == 1
    assert result["source"] == "fallback"
    assert result["rows"][0]["SK_ID_CURR"] == 54321
    assert "Lookup Case" in result["response"]


def test_run_pipeline_supports_compare_improvement_and_aggregate_queries(monkeypatch, tmp_path):
    db_path = init_database(tmp_path / "coverage.sqlite3")
    _seed_application(
        db_path,
        application_id=1,
        sk_id_curr=10001,
        applicant_name="Compare A",
        probability=0.10,
        decision="APPROVE",
        income=150000.0,
        credit=180000.0,
        ext1=0.8,
        ext2=0.78,
        ext3=0.76,
    )
    _seed_application(
        db_path,
        application_id=2,
        sk_id_curr=10002,
        applicant_name="Compare B",
        probability=0.42,
        decision="DECLINE",
        income=70000.0,
        credit=260000.0,
        ext1=0.35,
        ext2=0.4,
        ext3=0.38,
    )
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))

    compare_result = run_pipeline("Compare applicant 10001 and 10002")
    improvement_result = run_pipeline("How can applicant 10002 improve?")
    aggregate_result = run_pipeline("How many applicants were rejected?")

    assert compare_result["row_count"] == 2
    assert "10001" in compare_result["response"]
    assert "10002" in compare_result["response"]

    assert improvement_result["row_count"] == 1
    assert "10002" in improvement_result["response"]
    assert "reapply" in improvement_result["response"].lower()

    assert aggregate_result["row_count"] >= 1
    assert "applicants" in aggregate_result["response"].lower()


def test_chat_normalizes_pipeline_error_payloads_for_frontend(monkeypatch):
    monkeypatch.setattr(
        chatbot.agent_routes,
        "run_pipeline",
        lambda *args, **kwargs: {"error": "backend validation failed", "status_code": 400},
    )

    result = chatbot.chat("show applicant 99999")

    assert result["response"] == "backend validation failed"
    assert result["source"] == "fallback"
    assert "status_code" not in result


def test_run_pipeline_uses_loan_applications_as_chatbot_source(monkeypatch, tmp_path):
    db_path = init_database(tmp_path / "loan_source.sqlite3")
    _seed_application(
        db_path,
        application_id=1,
        sk_id_curr=77777,
        applicant_name="Loan Source Case",
        probability=0.10,
        decision="APPROVE",
        income=100000.0,
        credit=180000.0,
        ext1=0.7,
        ext2=0.72,
        ext3=0.74,
    )
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            UPDATE loan_applications
            SET last_probability = ?, last_decision = ?, last_model_version = ?
            WHERE id = ?
            """,
            (0.42, "DECLINE", "loan_apps_override_v1", 1),
        )
        conn.commit()

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))

    result = run_pipeline("show applicant 77777")

    assert result["row_count"] == 1
    assert result["rows"][0]["prediction"] == "DECLINE"
    assert result["rows"][0]["probability"] == 0.42
    assert result["rows"][0]["model_version"] == "loan_apps_override_v1"


def test_parse_query_supports_long_applicant_ids_for_decision_letters():
    parsed = query_parser.parse_query("Generate a decision letter for the applicant: 757906599")

    assert parsed.intent == query_parser.IntentType.DECISION_LETTER.value
    assert parsed.applicant_ids == [757906599]
