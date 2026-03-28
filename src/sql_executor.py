"""Read-only SQLite execution utilities for validated applicants queries."""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field

DB_PATH = os.environ.get("SQLITE_DB_PATH", "data/applicants.db")

logger = logging.getLogger(__name__)

_MODERN_TO_LEGACY_COLUMN_MAP = {
    "SK_ID_CURR": "applicant_id",
    "AMT_INCOME_TOTAL_CAPPED": "income",
    "AGE_YEARS": "age",
    "AMT_CREDIT": "mock_credit",
    "AMT_ANNUITY": "mock_annuity",
}


@dataclass
class ExecutionResult:
    """Structured result returned from query execution."""

    success: bool
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    error_message: str | None = None
    sql_executed: str = ""


def _resolve_db_path(db_path: str | None = None) -> str:
    return db_path or os.environ.get("SQLITE_DB_PATH", DB_PATH)


def _build_read_only_uri(db_path: str) -> str:
    normalized = os.path.abspath(db_path).replace("\\", "/")
    return f"file:{normalized}?mode=ro"


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a read-only SQLite connection configured for row dictionaries."""

    resolved_path = _resolve_db_path(db_path)
    connection = sqlite3.connect(
        _build_read_only_uri(resolved_path),
        uri=True,
        timeout=10.0,
    )
    connection.row_factory = sqlite3.Row
    return connection


def _row_to_plain_dict(row: sqlite3.Row) -> dict:
    plain: dict = {}
    for key in row.keys():
        value = row[key]
        if isinstance(value, (bytes, bytearray)):
            try:
                plain[key] = value.decode("utf-8", errors="replace")
            except Exception:
                plain[key] = str(value)
        else:
            plain[key] = value
    return plain


def _get_table_columns(connection: sqlite3.Connection, table_name: str = "applicants") -> set[str]:
    cursor = connection.cursor()
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        return {str(row[1]) for row in cursor.fetchall()}
    finally:
        cursor.close()


def _rewrite_sql_for_available_columns(sql: str, available_columns: set[str]) -> str:
    rewritten_sql = sql
    for modern_name, legacy_name in _MODERN_TO_LEGACY_COLUMN_MAP.items():
        if modern_name not in available_columns and legacy_name in available_columns:
            rewritten_sql = re.sub(
                rf"\b{re.escape(modern_name)}\b",
                legacy_name,
                rewritten_sql,
                flags=re.IGNORECASE,
            )

    if rewritten_sql != sql:
        logger.debug("[SQL_EXEC] Rewrote SQL for local schema compatibility: %s", rewritten_sql)

    return rewritten_sql


def execute_query(sql: str, db_path: str | None = None) -> ExecutionResult:
    """Execute a validated SQL query and return structured rows."""

    sql_text = "" if sql is None else str(sql).strip()
    if not sql_text:
        return ExecutionResult(
            success=False,
            error_message="Empty SQL query",
            sql_executed=sql_text,
        )

    connection: sqlite3.Connection | None = None
    cursor: sqlite3.Cursor | None = None

    try:
        connection = get_connection(db_path)
        sql_to_execute = _rewrite_sql_for_available_columns(sql_text, _get_table_columns(connection))
        cursor = connection.cursor()
        cursor.execute(sql_to_execute)

        columns = [description[0] for description in (cursor.description or [])]
        fetched_rows = cursor.fetchall()
        limited_rows = fetched_rows[:200]
        row_dicts = [_row_to_plain_dict(row) for row in limited_rows]

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
        return ExecutionResult(
            success=False,
            rows=[],
            row_count=0,
            columns=[],
            error_message=str(exc),
            sql_executed=sql_text,
        )
    except Exception as exc:
        logger.exception("Unexpected error while executing query: %s", exc)
        return ExecutionResult(
            success=False,
            rows=[],
            row_count=0,
            columns=[],
            error_message=str(exc),
            sql_executed=sql_text,
        )
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
    """Render execution results as a compact plain-text table."""

    if not result.success:
        return f"Query failed: {result.error_message or 'Unknown error'}"

    if result.row_count == 0:
        return "No matching applicants found."

    columns = result.columns or list(result.rows[0].keys())

    def _format_cell(value: object) -> str:
        text = "" if value is None else str(value)
        if len(text) > 50:
            return text[:47] + "..."
        return text

    rendered_rows = [
        {column: _format_cell(row.get(column, "")) for column in columns}
        for row in result.rows
    ]
    widths = [
        max(len(column), *(len(row[column]) for row in rendered_rows))
        for column in columns
    ]

    header = " | ".join(column.ljust(width) for column, width in zip(columns, widths))
    separator = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(row[column].ljust(width) for column, width in zip(columns, widths))
        for row in rendered_rows
    ]

    return "\n".join([header, separator, *body, f"Showing {result.row_count} result(s)"])


if __name__ == "__main__":
    live_db_path = _resolve_db_path()
    if os.path.exists(live_db_path):
        live_result = execute_query("SELECT * FROM applicants LIMIT 1;")
        print(format_results_as_text(live_result))
    else:
        print("DB not found, skipping live test.")
