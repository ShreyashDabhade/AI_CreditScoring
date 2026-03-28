"""Query execution engine for read-only applicants queries.

Provides:
- execute_query(sql_query: str, db_path: str) -> (list[dict], str)
- format_results_preview(results: list[dict], max_rows: int = 10) -> list[dict]

Behavior:
- Uses sqlite3.Row for named access
- Sets PRAGMA busy_timeout = 3000
- Limits fetched rows to 500
- All exceptions are caught and logged; function returns error codes
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = {}
    for k in row.keys():
        v = row[k]
        # convert bytes to str
        if isinstance(v, (bytes, bytearray)):
            try:
                d[k] = v.decode("utf-8", errors="replace")
            except Exception:
                d[k] = str(v)
        else:
            # sqlite3 may return numeric types; ensure JSON-serializable
            if v is None or isinstance(v, (int, float, str, bool)):
                d[k] = v
            else:
                # try to coerce to float, else str
                try:
                    d[k] = float(v)
                except Exception:
                    d[k] = str(v)
    return d


def execute_query(sql_query: str, db_path: str) -> Tuple[List[dict], str]:
    """Execute a validated read-only SQL query and return (results, status).

    Status values:
    - "OK": success
    - "NO_RESULTS": query returned no rows
    - "TIMEOUT": busy timeout / locked
    - "EXECUTION_ERROR: <msg>": other SQL execution error
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # set busy timeout (milliseconds)
        try:
            cur.execute("PRAGMA busy_timeout = 3000")
        except Exception:
            # ignore if pragma unsupported
            pass

        # execute
        cur.execute(sql_query)

        # fetch up to 500 rows
        rows = cur.fetchmany(500)
        results = [_row_to_dict(r) for r in rows]

        # close
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

        if not results:
            return [], "NO_RESULTS"
        return results, "OK"

    except sqlite3.OperationalError as e:
        msg = str(e)
        logger.exception("SQLite OperationalError executing query: %s", msg)
        if "locked" in msg.lower() or "timeout" in msg.lower():
            return [], "TIMEOUT"
        return [], f"EXECUTION_ERROR: {msg}"
    except sqlite3.DatabaseError as e:
        msg = str(e)
        logger.exception("SQLite DatabaseError executing query: %s", msg)
        return [], f"EXECUTION_ERROR: {msg}"
    except Exception as e:
        msg = str(e)
        logger.exception("Unexpected error executing query: %s", msg)
        return [], f"EXECUTION_ERROR: {msg}"


def _round_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (int, str, bool)):
        return v
    # try numeric coercion
    try:
        fv = float(v)
        return round(fv, 4)
    except Exception:
        return str(v)


def format_results_preview(results: List[dict], max_rows: int = 10) -> List[dict]:
    """Return first `max_rows` results with parsed `top_features` and rounded floats.

    - Parses `top_features` JSON strings back into lists when possible.
    - Rounds float values to 4 decimal places.
    """
    out = []
    for row in results[:max_rows]:
        r = {}
        for k, v in row.items():
            if k == "top_features" and isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    r[k] = parsed
                    continue
                except Exception:
                    # leave as original string if parse fails
                    pass
            r[k] = _round_value(v)
        out.append(r)
    return out
