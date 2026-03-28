from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


DEFAULT_DB_FILENAME = "mastermind_command_center.sqlite3"
ALLOWED_TIER_TYPES = frozenset({"FULL", "REDUCED"})
ALLOWED_STATUS_VALUES = frozenset(
    {
        "DRAFT",
        "SUBMITTED",
        "READY_FOR_REVIEW",
        "ANALYZED",
        "APPROVED",
        "REVIEW",
        "DECLINED",
    }
)
_UNSET = object()


def resolve_database_path(db_path: str | Path | None = None) -> str:
    if db_path is not None:
        return str(Path(db_path).resolve())
    project_root = Path(__file__).resolve().parents[1]
    return str((project_root / "data" / DEFAULT_DB_FILENAME).resolve())


def init_database(db_path: str | Path | None = None) -> str:
    resolved_path = Path(resolve_database_path(db_path))
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with _connect(resolved_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS loan_applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                applicant_name TEXT NOT NULL,
                sk_id_curr INTEGER NULL,
                submitted_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                tier_type TEXT NOT NULL,
                current_status TEXT NOT NULL,
                application_payload_json TEXT NOT NULL,
                last_probability REAL NULL,
                last_decision TEXT NULL,
                last_model_version TEXT NULL
            );

            CREATE TABLE IF NOT EXISTS score_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id INTEGER NOT NULL,
                scored_at TEXT NOT NULL,
                probability REAL NULL,
                decision TEXT NULL,
                model_version TEXT NULL,
                score_payload_json TEXT NOT NULL,
                FOREIGN KEY(application_id) REFERENCES loan_applications(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_loan_applications_status_updated
            ON loan_applications(current_status, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_score_runs_application_scored
            ON score_runs(application_id, scored_at DESC);

            CREATE TABLE IF NOT EXISTS application_change_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id INTEGER NOT NULL,
                changed_at TEXT NOT NULL,
                change_summary_json TEXT NOT NULL,
                FOREIGN KEY(application_id) REFERENCES loan_applications(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_application_change_log_application_changed
            ON application_change_log(application_id, changed_at DESC);
            """
        )

    return str(resolved_path)


def create_application(
    db_path: str | Path | None,
    *,
    applicant_name: str,
    tier_type: str,
    application_payload_json: Any,
    current_status: str = "DRAFT",
    sk_id_curr: int | None = None,
) -> dict[str, Any]:
    _validate_tier_type(tier_type)
    _validate_status(current_status)

    applicant_name = applicant_name.strip()
    if not applicant_name:
        raise ValueError("applicant_name must be non-empty")

    normalized_status = current_status.upper()
    timestamp = _utc_now()
    payload_text = _serialize_json(application_payload_json)

    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO loan_applications (
                applicant_name,
                sk_id_curr,
                submitted_at,
                updated_at,
                tier_type,
                current_status,
                application_payload_json,
                last_probability,
                last_decision,
                last_model_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
            """,
            (
                applicant_name,
                sk_id_curr,
                timestamp,
                timestamp,
                tier_type.upper(),
                normalized_status,
                payload_text,
            ),
        )
        application_id = int(cursor.lastrowid)

    application = get_application_by_id(db_path, application_id)
    if application is None:
        raise RuntimeError("Failed to reload created application")
    return application


def list_applications(
    db_path: str | Path | None,
    *,
    current_status: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT
            id,
            applicant_name,
            sk_id_curr,
            submitted_at,
            updated_at,
            tier_type,
            current_status,
            application_payload_json,
            last_probability,
            last_decision,
            last_model_version
        FROM loan_applications
    """
    params: list[Any] = []

    if current_status is not None:
        _validate_status(current_status)
        query += " WHERE current_status = ?"
        params.append(current_status)

    query += " ORDER BY updated_at DESC, id DESC"

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        query += " LIMIT ?"
        params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_application_by_id(
    db_path: str | Path | None,
    application_id: int,
) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                id,
                applicant_name,
                sk_id_curr,
                submitted_at,
                updated_at,
                tier_type,
                current_status,
                application_payload_json,
                last_probability,
                last_decision,
                last_model_version
            FROM loan_applications
            WHERE id = ?
            """,
            (application_id,),
        ).fetchone()
    return _row_to_dict(row) if row is not None else None


def update_application(
    db_path: str | Path | None,
    application_id: int,
    *,
    applicant_name: str | None | object = _UNSET,
    sk_id_curr: int | None | object = _UNSET,
    tier_type: str | object = _UNSET,
    current_status: str | object = _UNSET,
    application_payload_json: Any = _UNSET,
    last_probability: float | None | object = _UNSET,
    last_decision: str | None | object = _UNSET,
    last_model_version: str | None | object = _UNSET,
) -> dict[str, Any] | None:
    assignments: list[str] = []
    values: list[Any] = []

    if applicant_name is not _UNSET:
        if applicant_name is None or not str(applicant_name).strip():
            raise ValueError("applicant_name must be non-empty")
        assignments.append("applicant_name = ?")
        values.append(str(applicant_name).strip())

    if sk_id_curr is not _UNSET:
        assignments.append("sk_id_curr = ?")
        values.append(sk_id_curr)

    if tier_type is not _UNSET:
        normalized_tier = str(tier_type).upper()
        _validate_tier_type(normalized_tier)
        assignments.append("tier_type = ?")
        values.append(normalized_tier)

    if current_status is not _UNSET:
        normalized_status = str(current_status).upper()
        _validate_status(normalized_status)
        assignments.append("current_status = ?")
        values.append(normalized_status)

    if application_payload_json is not _UNSET:
        assignments.append("application_payload_json = ?")
        values.append(_serialize_json(application_payload_json))

    if last_probability is not _UNSET:
        assignments.append("last_probability = ?")
        values.append(last_probability)

    if last_decision is not _UNSET:
        assignments.append("last_decision = ?")
        values.append(last_decision)

    if last_model_version is not _UNSET:
        assignments.append("last_model_version = ?")
        values.append(last_model_version)

    if not assignments:
        return get_application_by_id(db_path, application_id)

    assignments.append("updated_at = ?")
    values.append(_utc_now())
    values.append(application_id)

    with _connect(db_path) as conn:
        cursor = conn.execute(
            f"""
            UPDATE loan_applications
            SET {", ".join(assignments)}
            WHERE id = ?
            """,
            tuple(values),
        )
        if cursor.rowcount == 0:
            return None

    return get_application_by_id(db_path, application_id)


def save_score_run(
    db_path: str | Path | None,
    *,
    application_id: int,
    probability: float | None,
    decision: str | None,
    model_version: str | None,
    score_payload_json: Any,
    current_status: str | None = None,
) -> dict[str, Any]:
    if get_application_by_id(db_path, application_id) is None:
        raise LookupError(f"Loan application not found: {application_id}")

    timestamp = _utc_now()
    payload_text = _serialize_json(score_payload_json)
    resolved_status = current_status or _status_from_decision(decision)
    _validate_status(resolved_status)

    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO score_runs (
                application_id,
                scored_at,
                probability,
                decision,
                model_version,
                score_payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                application_id,
                timestamp,
                probability,
                decision,
                model_version,
                payload_text,
            ),
        )
        conn.execute(
            """
            UPDATE loan_applications
            SET
                last_probability = ?,
                last_decision = ?,
                last_model_version = ?,
                current_status = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                probability,
                decision,
                model_version,
                resolved_status,
                timestamp,
                application_id,
            ),
        )
        score_run_id = int(cursor.lastrowid)

    score_run = _get_score_run_by_id(db_path, score_run_id)
    if score_run is None:
        raise RuntimeError("Failed to reload created score run")
    return score_run


def list_score_history_for_application(
    db_path: str | Path | None,
    application_id: int,
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                application_id,
                scored_at,
                probability,
                decision,
                model_version,
                score_payload_json
            FROM score_runs
            WHERE application_id = ?
            ORDER BY scored_at DESC, id DESC
            """,
            (application_id,),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def save_application_change_log(
    db_path: str | Path | None,
    *,
    application_id: int,
    change_summary_json: Any,
) -> dict[str, Any]:
    if get_application_by_id(db_path, application_id) is None:
        raise LookupError(f"Loan application not found: {application_id}")

    timestamp = _utc_now()
    payload_text = _serialize_json(change_summary_json)

    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO application_change_log (
                application_id,
                changed_at,
                change_summary_json
            )
            VALUES (?, ?, ?)
            """,
            (application_id, timestamp, payload_text),
        )
        change_log_id = int(cursor.lastrowid)

    change_log = _get_application_change_log_by_id(db_path, change_log_id)
    if change_log is None:
        raise RuntimeError("Failed to reload created application change log")
    return change_log


def list_application_change_history(
    db_path: str | Path | None,
    application_id: int,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT
            id,
            application_id,
            changed_at,
            change_summary_json
        FROM application_change_log
        WHERE application_id = ?
        ORDER BY changed_at DESC, id DESC
    """
    params: list[Any] = [application_id]

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        query += " LIMIT ?"
        params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [_row_to_dict(row) for row in rows]


def _get_score_run_by_id(
    db_path: str | Path | None,
    score_run_id: int,
) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                id,
                application_id,
                scored_at,
                probability,
                decision,
                model_version,
                score_payload_json
            FROM score_runs
            WHERE id = ?
            """,
            (score_run_id,),
        ).fetchone()
    return _row_to_dict(row) if row is not None else None


def _get_application_change_log_by_id(
    db_path: str | Path | None,
    change_log_id: int,
) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                id,
                application_id,
                changed_at,
                change_summary_json
            FROM application_change_log
            WHERE id = ?
            """,
            (change_log_id,),
        ).fetchone()
    return _row_to_dict(row) if row is not None else None


def _connect(db_path: str | Path | None) -> sqlite3.Connection:
    resolved_path = Path(resolve_database_path(db_path))
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(resolved_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _serialize_json(value: Any) -> str:
    if isinstance(value, str):
        json.loads(value)
        return value
    return json.dumps(value, sort_keys=True)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _validate_tier_type(tier_type: str) -> None:
    normalized = tier_type.upper()
    if normalized not in ALLOWED_TIER_TYPES:
        raise ValueError(f"tier_type must be one of {sorted(ALLOWED_TIER_TYPES)}")


def _validate_status(status: str) -> None:
    normalized = status.upper()
    if normalized not in ALLOWED_STATUS_VALUES:
        raise ValueError(f"current_status must be one of {sorted(ALLOWED_STATUS_VALUES)}")


def _status_from_decision(decision: str | None) -> str:
    normalized = str(decision or "").upper()
    if normalized == "APPROVE":
        return "APPROVED"
    if normalized == "DECLINE":
        return "DECLINED"
    if normalized == "REVIEW":
        return "REVIEW"
    return "ANALYZED"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "ALLOWED_STATUS_VALUES",
    "ALLOWED_TIER_TYPES",
    "DEFAULT_DB_FILENAME",
    "create_application",
    "get_application_by_id",
    "init_database",
    "list_applications",
    "list_application_change_history",
    "list_score_history_for_application",
    "resolve_database_path",
    "save_application_change_log",
    "save_score_run",
    "update_application",
]
