from __future__ import annotations

import os

from flask import Blueprint, request, jsonify

from src.query_parser import parse_query
from src.sql_executor import execute_query
from src.sql_generator import generate_sql
from src.sql_validator import validate_sql

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.route("/query", methods=["POST"])
def query_endpoint():
    """POST /analyst/query

    JSON body: { "query": "natural language query", "db_path": optional }
    Returns: { status: 'ok'|'error', rows: [...] } on success
    """
    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, dict):
        return jsonify({"status": "error", "reason": "invalid_json"}), 400

    user_query = payload.get("query")
    if not user_query or not isinstance(user_query, str):
        return jsonify({"status": "error", "reason": "missing query"}), 400

    db_path = payload.get("db_path") or os.environ.get("APPLICANTS_DB", "test_applicants.db")

    parsed = parse_query(user_query)
    sql = generate_sql(parsed, None)
    if not sql:
        return jsonify({"status": "error", "reason": "sql_generation_failed"}), 400

    validation = validate_sql(sql)
    if not validation.is_valid:
        return jsonify({"status": "error", "reason": validation.error_reason or "query_failed_or_unsafe"}), 400

    execution = execute_query(validation.cleaned_sql, db_path)
    if not execution.success:
        return jsonify({"status": "error", "reason": execution.error_message or "query_failed_or_unsafe"}), 400

    return jsonify(
        {
            "status": "ok",
            "rows": execution.rows,
            "row_count": execution.row_count,
            "columns": execution.columns,
            "sql": execution.sql_executed,
            "warnings": validation.warnings,
        }
    )
