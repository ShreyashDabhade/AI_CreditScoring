from __future__ import annotations

from flask import Flask

from src.api import create_app
from src.api import agent_routes
from src.query_parser import IntentType, ParsedQuery
from src.sql_executor import ExecutionResult
from src.sql_validator import ValidationResult


def _make_test_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(agent_routes.agent_bp)
    return app


def test_run_pipeline_handles_general_question(monkeypatch):
    parsed = ParsedQuery(
        intent=IntentType.GENERAL_QUESTION.value,
        applicant_ids=[],
        filters=[],
        columns_mentioned=[],
        raw_query="What is a credit score?",
        is_general=True,
    )

    monkeypatch.setattr(agent_routes.query_parser, "parse_query", lambda query: parsed)
    monkeypatch.setattr(
        agent_routes.response_generator,
        "handle_general_question",
        lambda query, model: "A credit score estimates repayment risk.",
    )

    result = agent_routes.run_pipeline("What is a credit score?")

    assert result == {
        "response": "A credit score estimates repayment risk.",
        "intent": IntentType.GENERAL_QUESTION.value,
        "sql": None,
        "rows": [],
        "row_count": 0,
    }


def test_run_pipeline_requires_two_ids_for_compare(monkeypatch):
    parsed = ParsedQuery(
        intent=IntentType.COMPARE_APPLICANTS.value,
        applicant_ids=[100002],
        filters=[],
        columns_mentioned=["applicant_id"],
        raw_query="Compare applicant 100002",
        is_general=False,
    )

    monkeypatch.setattr(agent_routes.query_parser, "parse_query", lambda query: parsed)

    result = agent_routes.run_pipeline("Compare applicant 100002")

    assert result == {
        "response": "Please provide two applicant IDs to compare. "
                    "Example: 'Compare applicant 100002 and 100003'",
        "intent": IntentType.COMPARE_APPLICANTS.value,
        "sql": None,
        "rows": [],
        "row_count": 0,
    }


def test_run_pipeline_uses_fallback_sql_and_trims_rows(monkeypatch):
    parsed = ParsedQuery(
        intent=IntentType.LOOKUP_APPLICANT.value,
        applicant_ids=[12345],
        filters=[],
        columns_mentioned=["applicant_id"],
        raw_query="show me applicant 12345",
        is_general=False,
    )
    rows = [{"applicant_id": 100000 + index} for index in range(12)]

    monkeypatch.setattr(agent_routes.query_parser, "parse_query", lambda query: parsed)

    def _raise_sql_error(parsed_query, model):
        raise RuntimeError("Gemini unavailable")

    monkeypatch.setattr(agent_routes.sql_generator, "generate_sql", _raise_sql_error)
    monkeypatch.setattr(
        agent_routes.sql_generator,
        "fallback_sql",
        lambda parsed_query: "SELECT * FROM applicants WHERE applicant_id = 12345 LIMIT 1",
    )
    monkeypatch.setattr(
        agent_routes.sql_validator,
        "validate_sql",
        lambda sql: ValidationResult(
            is_valid=True,
            cleaned_sql=f"{sql};",
            error_reason=None,
            warnings=[],
        ),
    )
    monkeypatch.setattr(
        agent_routes.sql_executor,
        "execute_query",
        lambda sql: ExecutionResult(
            success=True,
            rows=rows,
            row_count=len(rows),
            columns=["applicant_id"],
            error_message=None,
            sql_executed=sql,
        ),
    )
    monkeypatch.setattr(
        agent_routes.response_generator,
        "generate_response",
        lambda parsed_query, result, raw_query, model: "Applicant record found.",
    )

    result = agent_routes.run_pipeline("show me applicant 12345")

    assert result["response"] == "Applicant record found."
    assert result["intent"] == IntentType.LOOKUP_APPLICANT.value
    assert result["sql"] == "SELECT * FROM applicants WHERE applicant_id = 12345 LIMIT 1;"
    assert result["rows"] == rows[:10]
    assert result["row_count"] == 12


def test_run_pipeline_returns_validation_error_payload(monkeypatch):
    parsed = ParsedQuery(
        intent=IntentType.FILTER_APPLICANTS.value,
        applicant_ids=[],
        filters=["credit_score > 600"],
        columns_mentioned=["credit_score"],
        raw_query="show applicants with credit score > 600",
        is_general=False,
    )

    monkeypatch.setattr(agent_routes.query_parser, "parse_query", lambda query: parsed)
    monkeypatch.setattr(
        agent_routes.sql_generator,
        "generate_sql",
        lambda parsed_query, model: "SELECT bad_column FROM applicants",
    )
    monkeypatch.setattr(
        agent_routes.sql_validator,
        "validate_sql",
        lambda sql: ValidationResult(
            is_valid=False,
            cleaned_sql="SELECT bad_column FROM applicants;",
            error_reason="Potentially unsafe SQL",
            warnings=[],
        ),
    )

    result = agent_routes.run_pipeline("show applicants with credit score > 600")

    assert result["error"] == "Potentially unsafe SQL"
    assert result["intent"] == IntentType.FILTER_APPLICANTS.value
    assert result["status_code"] == 400


def test_run_pipeline_returns_helpful_message_when_no_sql_can_be_built(monkeypatch):
    parsed = ParsedQuery(
        intent=IntentType.AGGREGATE_QUERY.value,
        applicant_ids=[],
        filters=[],
        columns_mentioned=[],
        raw_query="what is the top rejection reason for applicants under 25?",
        is_general=False,
    )

    monkeypatch.setattr(agent_routes.query_parser, "parse_query", lambda query: parsed)
    monkeypatch.setattr(agent_routes.sql_generator, "generate_sql", lambda parsed_query, model: None)
    monkeypatch.setattr(agent_routes.sql_generator, "fallback_sql", lambda parsed_query: None)

    result = agent_routes.run_pipeline(parsed.raw_query)

    assert result["intent"] == IntentType.AGGREGATE_QUERY.value
    assert result["sql"] is None
    assert result["rows"] == []
    assert "couldn't build a database query" in result["response"]
    assert result["status_code"] == 200


def test_chat_endpoint_validates_non_empty_query():
    client = _make_test_app().test_client()

    response = client.post("/api/chat", json={"query": "   "})

    assert response.status_code == 400
    assert response.get_json() == {"error": "A non-empty 'query' field is required."}


def test_chat_endpoint_uses_pipeline_status_code(monkeypatch):
    client = _make_test_app().test_client()

    monkeypatch.setattr(
        agent_routes,
        "run_pipeline",
        lambda raw_query: {
            "error": "Query execution failed.",
            "intent": IntentType.FILTER_APPLICANTS.value,
            "sql": "SELECT * FROM applicants;",
            "rows": [],
            "row_count": 0,
            "status_code": 500,
        },
    )

    response = client.post("/api/chat", json={"query": "show high risk applicants"})

    assert response.status_code == 500
    assert response.get_json() == {
        "error": "Query execution failed.",
        "intent": IntentType.FILTER_APPLICANTS.value,
        "sql": "SELECT * FROM applicants;",
        "rows": [],
        "row_count": 0,
    }


def test_health_and_schema_endpoints_expose_expected_metadata(monkeypatch):
    monkeypatch.setattr(agent_routes, "get_model_name", lambda: "mock-gemini")
    client = _make_test_app().test_client()

    health_response = client.get("/api/health")
    schema_response = client.get("/api/schema")

    assert health_response.status_code == 200
    assert health_response.get_json()["status"] == "ok"
    assert health_response.get_json()["gemini_model"] == "mock-gemini"
    assert "db_path" in health_response.get_json()

    schema_payload = schema_response.get_json()
    assert schema_response.status_code == 200
    assert "applicant_id" in schema_payload["columns"]
    assert "Applicants table schema:" in schema_payload["schema"]


def test_create_app_registers_agent_api_routes(monkeypatch):
    monkeypatch.setattr(agent_routes, "get_model_name", lambda: "mock-gemini")
    client = create_app(mock_mode=True).test_client()

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
