from __future__ import annotations

import logging
from typing import Any

from gemini_client import get_model, is_available

from src import query_parser, response_generator, sql_executor, sql_generator, sql_validator

logger = logging.getLogger(__name__)


def _error_payload(
    message: str,
    *,
    intent: str = query_parser.IntentType.UNKNOWN.value,
    sql: str | None = None,
    status_code: int = 400,
    source: str = "fallback",
) -> dict[str, Any]:
    return {
        "error": message,
        "intent": intent,
        "sql": sql,
        "rows": [],
        "row_count": 0,
        "status_code": status_code,
        "source": source,
    }


def _active_source() -> str:
    return "gemini" if is_available() else "fallback"


def run_pipeline(
    raw_query: str,
    *,
    score_data: dict[str, Any] | None = None,
    shap_values: list[dict[str, Any]] | None = None,
    page_context: dict[str, Any] | None = None,
    report_context: dict[str, Any] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    context_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    del score_data, shap_values, page_context, report_context, conversation_history

    logger.info("Agent pipeline started")
    parsed = query_parser.parse_query(raw_query)
    logger.debug("Parsed query: %s", parsed)

    model = None
    try:
        model = get_model()
    except Exception:
        model = None

    if parsed.intent == query_parser.IntentType.GENERAL_QUESTION.value:
        return {
            "response": response_generator.handle_general_question(raw_query, model),
            "intent": parsed.intent,
            "sql": None,
            "rows": [],
            "row_count": 0,
            "source": _active_source(),
        }

    if parsed.intent == query_parser.IntentType.COMPARE_APPLICANTS.value and len(parsed.applicant_ids) < 2:
        return {
            "response": "Please provide two applicant IDs to compare. Example: 'Compare applicant 100002 and 100003'.",
            "intent": parsed.intent,
            "sql": None,
            "rows": [],
            "row_count": 0,
            "source": _active_source(),
        }

    try:
        sql = sql_generator.generate_sql(parsed, model)
    except Exception:
        logger.exception("SQL generation raised an exception")
        sql = None

    if not sql:
        return {
            "response": (
                "I understood your question but could not build a data query for it. "
                "Try rephrasing, for example: show applicant 100038, how many applicants were declined, or why was applicant 100038 declined?"
            ),
            "intent": parsed.intent,
            "sql": None,
            "rows": [],
            "row_count": 0,
            "status_code": 200,
            "source": _active_source(),
        }

    validation = sql_validator.validate_sql(sql)
    if not validation.is_valid:
        logger.error("SQL validation blocked query. SQL=%r validation=%s", sql, validation)
        return _error_payload(
            validation.error_reason or "SQL validation failed.",
            intent=parsed.intent,
            sql=validation.cleaned_sql or sql,
            status_code=400,
            source=_active_source(),
        )

    exec_result = sql_executor.execute_query(validation.cleaned_sql, context_rows=context_rows)
    if not exec_result.success:
        logger.error("SQL execution failed. SQL=%r validation=%s exec=%s", validation.cleaned_sql, validation, exec_result)
        return _error_payload(
            exec_result.error_message or "Query execution failed.",
            intent=parsed.intent,
            sql=validation.cleaned_sql,
            status_code=500,
            source=_active_source(),
        )

    response = response_generator.generate_response(parsed, exec_result, raw_query, model)
    return {
        "response": response,
        "intent": parsed.intent,
        "sql": validation.cleaned_sql,
        "rows": exec_result.rows[:10],
        "row_count": exec_result.row_count,
        "source": _active_source(),
    }


__all__ = ["run_pipeline"]
