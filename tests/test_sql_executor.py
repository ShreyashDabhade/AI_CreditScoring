import os
import sqlite3

from src.sql_executor import ExecutionResult, execute_query, format_results_as_text


def _build_test_db(db_path: str) -> None:
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE applicants (
            SK_ID_CURR INTEGER PRIMARY KEY,
            credit_score REAL,
            risk_band TEXT,
            explanation_text TEXT
        )
        """
    )
    cursor.executemany(
        "INSERT INTO applicants (SK_ID_CURR, credit_score, risk_band, explanation_text) VALUES (?, ?, ?, ?)",
        [
            (100001, 0.71, "low", "Stable payment history"),
            (100002, 0.42, "high", "High delinquency count"),
        ],
    )
    connection.commit()
    connection.close()


def test_execute_query_returns_structured_rows(tmp_path):
    db_path = tmp_path / "applicants.db"
    _build_test_db(str(db_path))

    result = execute_query(
        "SELECT SK_ID_CURR, credit_score FROM applicants ORDER BY SK_ID_CURR",
        str(db_path),
    )

    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.row_count == 2
    assert result.columns == ["SK_ID_CURR", "credit_score"]
    assert result.rows[0]["SK_ID_CURR"] == 100001
    assert result.sql_executed == "SELECT SK_ID_CURR, credit_score FROM applicants ORDER BY SK_ID_CURR"


def test_execute_query_returns_error_for_missing_db(tmp_path):
    missing_path = tmp_path / "missing.db"

    result = execute_query("SELECT * FROM applicants", str(missing_path))

    assert result.success is False
    assert result.rows == []
    assert result.row_count == 0
    assert result.error_message


def test_format_results_as_text_handles_success_empty_and_failure():
    success = ExecutionResult(
        success=True,
        rows=[{"SK_ID_CURR": 100001, "explanation_text": "x" * 60}],
        row_count=1,
        columns=["SK_ID_CURR", "explanation_text"],
        error_message=None,
        sql_executed="SELECT * FROM applicants",
    )
    empty = ExecutionResult(
        success=True,
        rows=[],
        row_count=0,
        columns=["SK_ID_CURR"],
        error_message=None,
        sql_executed="SELECT * FROM applicants WHERE 1 = 0",
    )
    failure = ExecutionResult(
        success=False,
        rows=[],
        row_count=0,
        columns=[],
        error_message="boom",
        sql_executed="SELECT * FROM applicants",
    )

    success_text = format_results_as_text(success)
    empty_text = format_results_as_text(empty)
    failure_text = format_results_as_text(failure)

    assert "SK_ID_CURR | explanation_text" in success_text
    assert "Showing 1 result(s)" in success_text
    assert "..." in success_text
    assert empty_text == "No matching applicants found."
    assert failure_text == "Query failed: boom"
