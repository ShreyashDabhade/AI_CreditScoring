from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

import os

from gemini_client import get_model_name

from src import (
    query_parser,
    response_generator,
    schema_map,
    sql_executor,
    sql_generator,
    sql_validator,
)

logger = logging.getLogger(__name__)

agent_bp = Blueprint("agent", __name__)

GEMINI_MODEL_NAME = "unavailable"
gemini_model = None


def _error_payload(
    message: str,
    *,
    intent: str = query_parser.IntentType.UNKNOWN.value,
    sql: str | None = None,
    status_code: int = 400,
) -> dict[str, Any]:
    return {
        "error": message,
        "intent": intent,
        "sql": sql,
        "rows": [],
        "row_count": 0,
        "status_code": status_code,
    }


def run_pipeline(raw_query: str) -> dict[str, Any]:
    """Run the full parse -> SQL -> validate -> execute -> respond pipeline."""
    global gemini_model, GEMINI_MODEL_NAME

    logger.info("Agent pipeline started")

    logger.info("Step 1/6: parsing user query")
    parsed = query_parser.parse_query(raw_query)
    logger.debug("Parsed query: %s", parsed)

    logger.info("Step 2/6: checking for general question intent")
    if parsed.intent == query_parser.IntentType.GENERAL_QUESTION.value:
        response = response_generator.handle_general_question(raw_query, gemini_model)
        return {
            "response": response,
            "intent": query_parser.IntentType.GENERAL_QUESTION.value,
            "sql": None,
            "rows": [],
            "row_count": 0,
        }

    if parsed.intent == query_parser.IntentType.COMPARE_APPLICANTS.value:
        if len(parsed.applicant_ids) < 2:
            return {
                "response": "Please provide two applicant IDs to compare. "
                            "Example: 'Compare applicant 100002 and 100003'",
                "intent": query_parser.IntentType.COMPARE_APPLICANTS.value,
                "sql": None,
                "rows": [],
                "row_count": 0,
            }

    logger.info("Step 3/6: generating SQL")
    sql: str | None = None
    validation = None
    exec_result = None
    fallback_candidate: str | None = None
    try:
        fallback_candidate = sql_generator.fallback_sql(parsed)
    except Exception:
        logger.exception("Fallback SQL generation raised an exception")
        fallback_candidate = None

    if parsed.intent == query_parser.IntentType.AGGREGATE_QUERY.value and fallback_candidate:
        logger.info("Using deterministic aggregate fallback SQL")
        sql = fallback_candidate

    try:
        if sql is None:
            sql = sql_generator.generate_sql(parsed, gemini_model)
    except Exception:
        logger.exception("Gemini SQL generation raised an exception")

    if not sql:
        logger.info("SQL generation returned no query; attempting fallback SQL")
        sql = fallback_candidate

    if not sql:
        logger.error(
            "[PIPELINE] Safe SQL fallback triggered. SQL was: %r | Validation result: %s | "
            "Execution result: %s | Parsed query: %s",
            sql,
            validation,
            exec_result,
            parsed,
        )
        return {
            "response": (
                "I understood your question but couldn't build a database query for it. "
                "Try rephrasing, for example:\n"
                "- show applicant 100038\n"
                "- how many applicants were rejected?\n"
                "- why was applicant 100038 rejected?"
            ),
            "intent": parsed.intent,
            "sql": None,
            "rows": [],
            "row_count": 0,
            "status_code": 200,
        }

    logger.info("Step 4/6: validating SQL")
    validation = sql_validator.validate_sql(sql)
    if not validation.is_valid:
        logger.error(
            "[PIPELINE] SQL validation blocked query. SQL was: %r | Validation result: %s",
            sql,
            validation,
        )
        return _error_payload(
            validation.error_reason or "SQL validation failed.",
            intent=parsed.intent,
            sql=validation.cleaned_sql or sql,
            status_code=400,
        )

    for warning in validation.warnings:
        logger.warning("SQL validation warning: %s", warning)

    logger.info("Step 5/6: executing SQL")
    exec_result = sql_executor.execute_query(validation.cleaned_sql)
    if not exec_result.success:
        logger.error(
            "[PIPELINE] SQL execution failed. SQL was: %r | Validation result: %s | Execution result: %s",
            validation.cleaned_sql,
            validation,
            exec_result,
        )
        return _error_payload(
            exec_result.error_message or "Query execution failed.",
            intent=parsed.intent,
            sql=validation.cleaned_sql,
            status_code=500,
        )

    logger.info("Step 6/6: generating natural-language response")
    response = response_generator.generate_response(parsed, exec_result, raw_query, gemini_model)

    logger.info("Agent pipeline completed successfully")
    return {
        "response": response,
        "intent": parsed.intent,
        "sql": validation.cleaned_sql,
        "rows": exec_result.rows[:10],
        "row_count": exec_result.row_count,
    }


@agent_bp.post("/api/chat")
def chat_endpoint():
    """Chatbot endpoint for the frontend."""

    try:
        payload = request.get_json(silent=True)
        query = "" if not isinstance(payload, dict) else str(payload.get("query", "")).strip()

        if not query:
            return jsonify({"error": "A non-empty 'query' field is required."}), 400

        response_payload = run_pipeline(query)
        status_code = int(response_payload.pop("status_code", 200))
        return jsonify(response_payload), status_code
    except Exception:
        logger.exception("Unhandled error in /api/chat")
        return jsonify({"error": "Internal server error."}), 500


@agent_bp.get("/api/health")
def health_endpoint():
    """Lightweight health endpoint for the chatbot routes."""

    try:
        model_name = get_model_name()
        return jsonify(
            {
                "status": "ok",
                "gemini_model": model_name,
                "db_path": os.getenv("SQLITE_DB_PATH", "data/applicants.db"),
            }
        )
    except Exception:
        logger.exception("Unhandled error in /api/health")
        return jsonify({"error": "Internal server error."}), 500


@agent_bp.get("/api/schema")
def schema_endpoint():
    """Expose schema metadata used by the chatbot pipeline."""

    try:
        return jsonify(
            {
                "columns": schema_map.get_all_column_names(),
                "schema": schema_map.get_schema_prompt(),
            }
        )
    except Exception:
        logger.exception("Unhandled error in /api/schema")
        return jsonify({"error": "Internal server error."}), 500


__all__ = [
    "GEMINI_MODEL_NAME",
    "agent_bp",
    "gemini_model",
    "run_pipeline",
]
