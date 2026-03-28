"""Centralized error handling and logging for the credit analyst system.

Provides structured error codes and helpers to log and return safe error
responses. The module configures a file logger on import and respects
DEBUG_MODE environment variable for including technical details.
"""
from __future__ import annotations

import json
import logging
import os
import traceback
from typing import Any, Callable, Dict, Optional, Tuple


# Error codes
INVALID_QUERY = "INVALID_QUERY"
NO_RESULTS = "NO_RESULTS"
UNSAFE_SQL = "UNSAFE_SQL"
SQL_GENERATION_FAILED = "SQL_GENERATION_FAILED"
EXECUTION_ERROR = "EXECUTION_ERROR"
AGENT_ERROR = "AGENT_ERROR"
SCHEMA_ERROR = "SCHEMA_ERROR"
DB_ERROR = "DB_ERROR"


# Logging setup
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "credit_agent.log")

DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() in ("1", "true", "yes")

logger = logging.getLogger("credit_agent.error_handler")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

# File handler
fh = logging.FileHandler(LOG_PATH)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(module)s | %(message)s")
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)

# Console handler in debug mode
if DEBUG_MODE:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def _user_message_for_code(code: str) -> str:
    mapping = {
        INVALID_QUERY: "The query was invalid.",
        NO_RESULTS: "No results were found for the request.",
        UNSAFE_SQL: "The generated SQL was blocked for safety reasons.",
        SQL_GENERATION_FAILED: "Could not generate SQL from the request.",
        EXECUTION_ERROR: "An error occurred while executing the query.",
        AGENT_ERROR: "The language model failed to produce a valid response.",
        SCHEMA_ERROR: "Database schema is incompatible or missing required columns.",
        DB_ERROR: "A database error occurred.",
    }
    return mapping.get(code, "An internal error occurred.")


def handle_error(error_code: str, detail: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Log the error and return a structured error response dict.

    - `detail` is included in returned dict only when DEBUG_MODE is true.
    - Logs at WARNING for expected issues, ERROR for serious failures.
    """
    ctx = context or {}
    module = ctx.get("module") or "unknown"

    user_msg = _user_message_for_code(error_code)

    # choose log level
    serious = error_code in {EXECUTION_ERROR, AGENT_ERROR, DB_ERROR}

    log_msg = f"code={error_code} module={module} detail={detail} context={json.dumps(ctx, default=str)}"
    if serious:
        logger.error(log_msg)
    else:
        logger.warning(log_msg)

    # include stack trace in debug mode
    tech_detail = detail
    if DEBUG_MODE:
        try:
            tech_detail = f"{detail} | traceback: {traceback.format_exc()}"
        except Exception:
            tech_detail = detail
    else:
        tech_detail = "" if detail is None else ""

    return {
        "status": "error",
        "error_code": error_code,
        "message": user_msg,
        "detail": tech_detail,
        "module": module,
    }


def safe_execute(fn: Callable, error_code: str, context: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Execute `fn(*args, **kwargs)` and return (result, None) on success.

    On exception, returns (None, error_dict) where error_dict is produced by
    `handle_error`. The caller should not see raw exceptions.
    """
    try:
        res = fn(*args, **kwargs)
        return res, None
    except Exception as e:
        # classify common exceptions if possible
        msg = str(e)
        ctx = context or {}

        # map some exception messages to codes
        try:
            import sqlite3 as _sqlite

            is_sqlite_err = isinstance(e, _sqlite.DatabaseError)
        except Exception:
            is_sqlite_err = False

        if is_sqlite_err:
            code = DB_ERROR
        else:
            code = error_code or EXECUTION_ERROR

        err = handle_error(code, msg, ctx)
        return None, err
