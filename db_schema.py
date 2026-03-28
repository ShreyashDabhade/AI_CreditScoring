"""SQLite schema builder for applicants table.

Provides:
- sanitize_column_name(name: str) -> str
- get_column_names(feature_names: list[str]) -> list[str]
- create_schema(db_path: str, feature_names: list[str]) -> None

This module uses only the stdlib `sqlite3` and builds a table with fixed columns
first followed by dynamic feature columns (REAL). Column names are sanitized to
be safe for SQLite and idempotent.
"""
from __future__ import annotations

import re
import sqlite3
from typing import List


# Fixed columns (ordered)
FIXED_COLUMNS = [
    ("applicant_id", "INTEGER PRIMARY KEY"),
    ("income", "REAL"),
    ("age", "REAL"),
    ("probability", "REAL"),
    ("prediction", "INTEGER"),
    ("credit_score", "INTEGER"),
    ("risk_band", "TEXT"),
    ("top_features", "TEXT"),  # JSON string
    ("explanation_text", "TEXT"),
]


_SAFE_RE = re.compile(r"[^a-z0-9_]")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def sanitize_column_name(name: str) -> str:
    """Sanitize a feature name into a valid SQLite column name.

    Rules:
    - Lowercase
    - Replace any non [a-z0-9_] with underscore
    - Collapse multiple underscores
    - Strip leading/trailing underscores
    - Strip leading digits
    - If result is empty, return 'col'

    Idempotent: applying this function multiple times yields the same result.
    """
    if not isinstance(name, str):
        name = str(name)
    s = name.lower()
    s = _SAFE_RE.sub("_", s)
    s = _MULTI_UNDERSCORE_RE.sub("_", s)
    s = s.strip("_")
    # strip leading digits
    s = re.sub(r"^[0-9]+", "", s)
    s = s.strip("_")
    if not s:
        return "col"
    return s


def get_column_names(feature_names: List[str]) -> List[str]:
    """Return ordered list of all column names: fixed columns first, then sanitized features.

    Ensures uniqueness: if a sanitized feature collides with an existing fixed
    or previously-seen column name, a numeric suffix is appended to make it unique.
    """
    cols = [name for name, _ in FIXED_COLUMNS]
    seen = set(cols)
    for orig in feature_names:
        cand = sanitize_column_name(orig)
        base = cand
        i = 1
        while cand in seen:
            cand = f"{base}_{i}"
            i += 1
        seen.add(cand)
        cols.append(cand)
    return cols


def create_schema(db_path: str, feature_names: List[str]) -> None:
    """Create `applicants` table in SQLite database at `db_path`.

    The table will contain fixed columns (defined first) followed by dynamic
    REAL columns for each sanitized feature name. Indexes are created for
    common access patterns.
    """
    columns = get_column_names(feature_names)

    # build column definitions preserving order: fixed then dynamic
    fixed_defs = [f"{name} {dtype}" for name, dtype in FIXED_COLUMNS]
    # dynamic names correspond to columns[len(fixed_defs):]
    dynamic_names = columns[len(fixed_defs) :]
    dynamic_defs = [f"{name} REAL" for name in dynamic_names]

    create_stmt = (
        "CREATE TABLE IF NOT EXISTS applicants ("
        + ", ".join(fixed_defs + dynamic_defs)
        + ")"
    )

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(create_stmt)
        # create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_income ON applicants(income)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_age ON applicants(age)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_applicant_id ON applicants(applicant_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_risk_band ON applicants(risk_band)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction ON applicants(prediction)")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    # simple self-test when run directly
    sample_features = [f"feature {i}" for i in range(1, 6)]
    create_schema("test_applicants.db", sample_features)
    print("Created test_applicants.db with sample schema")
