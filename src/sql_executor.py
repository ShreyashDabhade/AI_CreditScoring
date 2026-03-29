"""Read-only execution utilities for the active chatbot pipeline."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from src.db_manager import resolve_database_path

DB_PATH = os.environ.get("SQLITE_DB_PATH", resolve_database_path())

logger = logging.getLogger(__name__)

_CONTEXT_COLUMNS = [
    "application_id",
    "SK_ID_CURR",
    "applicant_id",
    "applicant_name",
    "AMT_INCOME_TOTAL_CAPPED",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AGE_YEARS",
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "CNT_FAM_MEMBERS",
    "DAYS_EMPLOYED",
    "prediction",
    "credit_score",
    "risk_band",
    "probability",
    "top_features",
    "explanation_text",
    "CREDIT_INCOME_RATIO",
    "EXT_SOURCE_MEAN",
    "INST_LATE_COUNT",
    "coverage_tier",
    "model_version",
]


@dataclass
class ExecutionResult:
    success: bool
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    error_message: str | None = None
    sql_executed: str = ""


def _resolve_db_path(db_path: str | None = None) -> str:
    return str(db_path or os.environ.get("SQLITE_DB_PATH") or DB_PATH)


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    connection = sqlite3.connect(_resolve_db_path(db_path), timeout=10.0)
    connection.row_factory = sqlite3.Row
    return connection


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _row_to_plain_dict(row: sqlite3.Row) -> dict:
    plain: dict = {}
    for key in row.keys():
        value = row[key]
        if key in {"top_features"}:
            plain[key] = _safe_json_loads(value)
        else:
            plain[key] = value
    return plain


def _normalize_sql(sql: str) -> str:
    return str(sql or "").strip().rstrip(";")


def _command_center_query(sql: str) -> str:
    core_sql = _normalize_sql(sql)
    return (
        "WITH applicants AS ("
        "    SELECT "
        "        la.id AS application_id, "
        "        COALESCE(la.sk_id_curr, json_extract(la.application_payload_json, '$.application.SK_ID_CURR'), la.id) AS SK_ID_CURR, "
        "        COALESCE(la.sk_id_curr, json_extract(la.application_payload_json, '$.application.SK_ID_CURR'), la.id) AS applicant_id, "
        "        la.applicant_name AS applicant_name, "
        "        CAST(json_extract(la.application_payload_json, '$.application.AMT_INCOME_TOTAL_CAPPED') AS REAL) AS AMT_INCOME_TOTAL_CAPPED, "
        "        CAST(json_extract(la.application_payload_json, '$.application.AMT_CREDIT') AS REAL) AS AMT_CREDIT, "
        "        CAST(json_extract(la.application_payload_json, '$.application.AMT_ANNUITY') AS REAL) AS AMT_ANNUITY, "
        "        CAST(COALESCE(json_extract(la.application_payload_json, '$.application.AGE_YEARS'), "
        "             ABS(CAST(json_extract(la.application_payload_json, '$.application.DAYS_BIRTH') AS REAL)) / 365.25) AS REAL) AS AGE_YEARS, "
        "        json_extract(la.application_payload_json, '$.application.CODE_GENDER') AS CODE_GENDER, "
        "        json_extract(la.application_payload_json, '$.application.NAME_EDUCATION_TYPE') AS NAME_EDUCATION_TYPE, "
        "        CAST(json_extract(la.application_payload_json, '$.application.CNT_FAM_MEMBERS') AS REAL) AS CNT_FAM_MEMBERS, "
        "        CAST(json_extract(la.application_payload_json, '$.application.DAYS_EMPLOYED') AS REAL) AS DAYS_EMPLOYED, "
        "        UPPER(COALESCE(la.last_decision, 'UNKNOWN')) AS prediction, "
        "        CAST(ROUND(300 + ((1 - COALESCE(la.last_probability, 0.5)) * 550), 0) AS REAL) AS credit_score, "
        "        CASE "
        "            WHEN COALESCE(la.last_probability, 1.0) < 0.15 THEN 'LOW' "
        "            WHEN COALESCE(la.last_probability, 1.0) < 0.35 THEN 'MEDIUM' "
        "            ELSE 'HIGH' "
        "        END AS risk_band, "
        "        CAST(la.last_probability AS REAL) AS probability, "
        "        json_extract(la.application_payload_json, '$.application.top_features') AS top_features, "
        "        COALESCE(json_extract(la.application_payload_json, '$.application.explanation_text'), la.last_decision, 'No saved analysis details are available.') AS explanation_text, "
        "        CASE "
        "            WHEN CAST(json_extract(la.application_payload_json, '$.application.AMT_INCOME_TOTAL_CAPPED') AS REAL) IS NULL "
        "              OR CAST(json_extract(la.application_payload_json, '$.application.AMT_INCOME_TOTAL_CAPPED') AS REAL) = 0 THEN NULL "
        "            ELSE CAST(json_extract(la.application_payload_json, '$.application.AMT_CREDIT') AS REAL) "
        "              / CAST(json_extract(la.application_payload_json, '$.application.AMT_INCOME_TOTAL_CAPPED') AS REAL) "
        "        END AS CREDIT_INCOME_RATIO, "
        "        ("
        "            COALESCE(CAST(json_extract(la.application_payload_json, '$.application.EXT_SOURCE_1') AS REAL), 0) + "
        "            COALESCE(CAST(json_extract(la.application_payload_json, '$.application.EXT_SOURCE_2') AS REAL), 0) + "
        "            COALESCE(CAST(json_extract(la.application_payload_json, '$.application.EXT_SOURCE_3') AS REAL), 0)"
        "        ) / NULLIF("
        "            CASE WHEN json_extract(la.application_payload_json, '$.application.EXT_SOURCE_1') IS NOT NULL THEN 1 ELSE 0 END + "
        "            CASE WHEN json_extract(la.application_payload_json, '$.application.EXT_SOURCE_2') IS NOT NULL THEN 1 ELSE 0 END + "
        "            CASE WHEN json_extract(la.application_payload_json, '$.application.EXT_SOURCE_3') IS NOT NULL THEN 1 ELSE 0 END, 0"
        "        ) AS EXT_SOURCE_MEAN, "
        "        CAST(COALESCE(json_extract(la.application_payload_json, '$.application.INST_LATE_COUNT'), 0) AS REAL) AS INST_LATE_COUNT, "
        "        UPPER(COALESCE(la.tier_type, 'REDUCED')) AS coverage_tier, "
        "        la.last_model_version AS model_version "
        "    FROM loan_applications la "
        ") "
        f"{core_sql};"
    )


def _create_context_table(connection: sqlite3.Connection, context_rows: list[dict[str, Any]]) -> None:
    connection.execute(
        """
        CREATE TABLE applicants (
            application_id REAL,
            SK_ID_CURR REAL,
            applicant_id REAL,
            applicant_name TEXT,
            AMT_INCOME_TOTAL_CAPPED REAL,
            AMT_CREDIT REAL,
            AMT_ANNUITY REAL,
            AGE_YEARS REAL,
            CODE_GENDER TEXT,
            NAME_EDUCATION_TYPE TEXT,
            CNT_FAM_MEMBERS REAL,
            DAYS_EMPLOYED REAL,
            prediction TEXT,
            credit_score REAL,
            risk_band TEXT,
            probability REAL,
            top_features TEXT,
            explanation_text TEXT,
            CREDIT_INCOME_RATIO REAL,
            EXT_SOURCE_MEAN REAL,
            INST_LATE_COUNT REAL,
            coverage_tier TEXT,
            model_version TEXT
        )
        """
    )
    insert_sql = (
        "INSERT INTO applicants ("
        + ", ".join(_CONTEXT_COLUMNS)
        + ") VALUES ("
        + ", ".join("?" for _ in _CONTEXT_COLUMNS)
        + ")"
    )
    for row in context_rows:
        values: list[Any] = []
        for column in _CONTEXT_COLUMNS:
            value = row.get(column)
            if column == "top_features" and not isinstance(value, str):
                value = json.dumps(value or [])
            values.append(value)
        connection.execute(insert_sql, tuple(values))
    connection.commit()


def execute_query(
    sql: str,
    db_path: str | None = None,
    context_rows: list[dict[str, Any]] | None = None,
) -> ExecutionResult:
    sql_text = _normalize_sql(sql)
    if not sql_text:
        return ExecutionResult(False, error_message="Empty SQL query", sql_executed=sql_text)

    connection: sqlite3.Connection | None = None
    cursor: sqlite3.Cursor | None = None
    sql_to_execute = sql_text

    try:
        if context_rows:
            connection = sqlite3.connect(":memory:", timeout=10.0)
            connection.row_factory = sqlite3.Row
            _create_context_table(connection, context_rows)
        else:
            connection = get_connection(db_path)
            sql_to_execute = _command_center_query(sql_text)

        cursor = connection.cursor()
        cursor.execute(sql_to_execute)
        columns = [description[0] for description in (cursor.description or [])]
        fetched_rows = cursor.fetchall()
        row_dicts = [_row_to_plain_dict(row) for row in fetched_rows[:200]]
        return ExecutionResult(
            success=True,
            rows=row_dicts,
            row_count=len(row_dicts),
            columns=columns,
            error_message=None,
            sql_executed=sql_to_execute,
        )
    except sqlite3.Error as exc:
        logger.exception("SQLite error while executing query: %s", exc)
        return ExecutionResult(False, rows=[], row_count=0, columns=[], error_message=str(exc), sql_executed=sql_to_execute)
    except Exception as exc:
        logger.exception("Unexpected error while executing query: %s", exc)
        return ExecutionResult(False, rows=[], row_count=0, columns=[], error_message=str(exc), sql_executed=sql_to_execute)
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def format_results_as_text(result: ExecutionResult) -> str:
    if not result.success:
        return f"Query failed: {result.error_message or 'Unknown error'}"
    if result.row_count == 0:
        return "No matching applicants found."
    columns = result.columns or list(result.rows[0].keys())

    def _format_cell(value: object) -> str:
        text = "" if value is None else str(value)
        return text if len(text) <= 50 else text[:47] + "..."

    rendered_rows = [{column: _format_cell(row.get(column, "")) for column in columns} for row in result.rows]
    widths = [max(len(column), *(len(row[column]) for row in rendered_rows)) for column in columns]
    header = " | ".join(column.ljust(width) for column, width in zip(columns, widths))
    separator = "-+-".join("-" * width for width in widths)
    body = [" | ".join(row[column].ljust(width) for column, width in zip(columns, widths)) for row in rendered_rows]
    return "\n".join([header, separator, *body, f"Showing {result.row_count} result(s)"])
