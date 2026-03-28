"""SQL validation and sanitization utilities for applicants queries.

This module performs lightweight defensive checks on AI-generated SQL before
execution. It is intentionally regex-based and depends only on ``schema_map``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, List

try:
    from . import schema_map
except ImportError:  # pragma: no cover - allows `python src/sql_validator.py`
    import schema_map  # type: ignore


@dataclass
class ValidationResult:
    """Outcome of SQL validation and sanitization."""

    is_valid: bool
    cleaned_sql: str
    error_reason: str | None
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allow simple truthiness checks in older call sites."""
        return self.is_valid

    def __iter__(self) -> Iterator[object]:
        """Allow unpacking into ``(ok, reason)`` for backward compatibility."""
        yield self.is_valid
        yield self.error_reason or "OK"


_FORBIDDEN_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "EXEC",
)

_SQL_FUNCTIONS = {
    "ABS",
    "AVG",
    "CAST",
    "COALESCE",
    "COUNT",
    "DATE",
    "DATETIME",
    "LOWER",
    "MAX",
    "MIN",
    "ROUND",
    "STRFTIME",
    "SUBSTR",
    "SUM",
    "UPPER",
}

_SQL_KEYWORDS = {
    "AND",
    "AS",
    "ASC",
    "BETWEEN",
    "BY",
    "CASE",
    "DESC",
    "DISTINCT",
    "ELSE",
    "END",
    "FROM",
    "GROUP",
    "HAVING",
    "IN",
    "IS",
    "JOIN",
    "LIKE",
    "LIMIT",
    "NOT",
    "NULL",
    "ON",
    "OR",
    "ORDER",
    "SELECT",
    "THEN",
    "TRUE",
    "WHEN",
    "WHERE",
    "FALSE",
}

_NON_COLUMN_TOKENS = {
    ">",
    "<",
    "=",
    ">=",
    "<=",
    "!=",
    "BETWEEN",
    "AND",
    "OR",
    "NOT",
    "IN",
    "LIKE",
    "NULL",
    "TRUE",
    "FALSE",
}

_VALID_COLUMNS_BY_LOWER = {
    column.lower(): column
    for column in (
        schema_map.get_all_column_names() if hasattr(schema_map, "get_all_column_names") else []
    )
}


def _strip_string_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'", "''", sql)


def _normalize_identifier(identifier: str) -> str:
    token = identifier.strip().strip("[]`\"")
    if "." in token:
        token = token.split(".")[-1]
    return token


def _unique_in_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _split_top_level_csv(section: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0

    for char in section:
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1

        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)

    return parts


def _extract_identifiers(expression: str) -> List[str]:
    cleaned = _strip_string_literals(expression)
    cleaned = re.sub(
        r"\s+AS\s+[A-Za-z_][A-Za-z0-9_]*\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    identifiers: List[str] = []
    for match in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?\b",
        cleaned,
    ):
        token = _normalize_identifier(match.group(0))
        upper_token = token.upper()
        if upper_token in _SQL_KEYWORDS or upper_token in _SQL_FUNCTIONS:
            continue
        if upper_token in _NON_COLUMN_TOKENS:
            continue
        if len(token) < 2:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            continue
        if re.fullmatch(r"(?:>=|<=|!=|=|>|<)+", token):
            continue
        if (token.startswith("'") and token.endswith("'")) or (token.startswith('"') and token.endswith('"')):
            continue
        if token.lower() == "applicants":
            continue
        identifiers.append(token)

    return identifiers


def check_forbidden_keywords(sql: str) -> str | None:
    """Return the first forbidden keyword found, if any."""

    text = "" if sql is None else str(sql)
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE):
            return keyword
    return None


def check_table_name(sql: str) -> bool:
    """Return True when the SQL references only the ``applicants`` table."""

    cleaned = sanitize_sql(sql)
    matches = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_\.]*)", cleaned, flags=re.IGNORECASE)
    if not matches:
        return False

    normalized = [_normalize_identifier(name).lower() for name in matches]
    return all(name == "applicants" for name in normalized)


def extract_columns_from_sql(sql: str) -> List[str]:
    """Extract likely column names referenced in SQL clauses."""

    cleaned = sanitize_sql(sql)
    stripped = _strip_string_literals(cleaned)
    columns: List[str] = []

    select_match = re.search(r"\bSELECT\b(.*?)\bFROM\b", stripped, flags=re.IGNORECASE | re.DOTALL)
    if select_match:
        for expression in _split_top_level_csv(select_match.group(1)):
            if expression.strip() == "*":
                columns.append("*")
            else:
                columns.extend(_extract_identifiers(expression))

    where_match = re.search(
        r"\bWHERE\b(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|;|$)",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if where_match:
        columns.extend(_extract_identifiers(where_match.group(1)))

    order_match = re.search(
        r"\bORDER\s+BY\b(.*?)(?:\bLIMIT\b|;|$)",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if order_match:
        for expression in _split_top_level_csv(order_match.group(1)):
            expression = re.sub(r"\bASC\b|\bDESC\b", "", expression, flags=re.IGNORECASE)
            columns.extend(_extract_identifiers(expression))

    group_match = re.search(
        r"\bGROUP\s+BY\b(.*?)(?:\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|;|$)",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if group_match:
        for expression in _split_top_level_csv(group_match.group(1)):
            columns.extend(_extract_identifiers(expression))

    having_match = re.search(
        r"\bHAVING\b(.*?)(?:\bORDER\s+BY\b|\bLIMIT\b|;|$)",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if having_match:
        columns.extend(_extract_identifiers(having_match.group(1)))

    return _unique_in_order(columns)


def check_column_names(sql: str) -> List[str]:
    """Return any referenced column names that are not in the known schema."""

    invalid_columns: List[str] = []
    for column_name in extract_columns_from_sql(sql):
        if not column_name or column_name == "*":
            continue

        upper_name = column_name.upper()
        if upper_name in _SQL_FUNCTIONS or upper_name in _SQL_KEYWORDS:
            continue
        if upper_name in _NON_COLUMN_TOKENS:
            continue
        if len(column_name) < 2:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", column_name):
            continue
        if re.fullmatch(r"(?:>=|<=|!=|=|>|<)+", column_name):
            continue
        if column_name.startswith("'") and column_name.endswith("'"):
            continue

        if schema_map.is_valid_column(column_name):
            continue

        if column_name.lower() in _VALID_COLUMNS_BY_LOWER:
            continue

        invalid_columns.append(column_name)

    return _unique_in_order(invalid_columns)


def sanitize_sql(sql: str) -> str:
    """Remove markdown fences, normalize whitespace, and end with a semicolon."""

    if sql is None:
        return ""

    cleaned = str(sql)
    cleaned = re.sub(r"```(?:sql)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"^\s*sql\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return ""

    cleaned = cleaned.rstrip(";").strip()
    return f"{cleaned};"


def validate_sql(sql: str) -> ValidationResult:
    """Sanitize SQL and validate safety, table usage, and column names."""

    cleaned = sanitize_sql(sql)
    if not cleaned:
        return ValidationResult(
            is_valid=False,
            cleaned_sql="",
            error_reason="Empty query",
            warnings=[],
        )

    if len(cleaned) > 2000:
        return ValidationResult(
            is_valid=False,
            cleaned_sql=cleaned,
            error_reason="Query exceeds maximum allowed length (2000 characters)",
            warnings=[],
        )

    if "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        return ValidationResult(
            is_valid=False,
            cleaned_sql=cleaned,
            error_reason="Comments are not allowed in SQL",
            warnings=[],
        )

    if not re.match(r"^\s*SELECT\b", cleaned, flags=re.IGNORECASE):
        return ValidationResult(
            is_valid=False,
            cleaned_sql=cleaned,
            error_reason="Only SELECT statements are allowed",
            warnings=[],
        )

    forbidden = check_forbidden_keywords(cleaned)
    if forbidden:
        return ValidationResult(
            is_valid=False,
            cleaned_sql=cleaned,
            error_reason=f"Forbidden SQL keyword detected: {forbidden}",
            warnings=[],
        )

    if ";" in cleaned[:-1]:
        return ValidationResult(
            is_valid=False,
            cleaned_sql=cleaned,
            error_reason="Only a single SQL statement is allowed",
            warnings=[],
        )

    if not check_table_name(cleaned):
        return ValidationResult(
            is_valid=False,
            cleaned_sql=cleaned,
            error_reason="Query must reference only the applicants table",
            warnings=[],
        )

    warnings: List[str] = []
    invalid_columns = check_column_names(cleaned)
    if invalid_columns:
        warnings.append(
            "Potentially invalid column names: " + ", ".join(invalid_columns)
        )

    return ValidationResult(
        is_valid=True,
        cleaned_sql=cleaned,
        error_reason=None,
        warnings=warnings,
    )


def is_explanation_query(sql: str) -> bool:
    """Return True when explanation-related fields are selected."""

    cleaned = sanitize_sql(sql).lower()
    return "top_features" in cleaned or "explanation_text" in cleaned


if __name__ == "__main__":
    sample_sql = [
        "SELECT applicant_id, credit_score FROM applicants WHERE risk_band = 'LOW'",
        "SELECT * FROM applicants; DROP TABLE applicants;",
        "SELECT made_up_column FROM applicants WHERE applicant_id = 12345",
        "```sql\nSELECT * FROM applicants WHERE applicant_id = 12345\n```",
    ]

    for query in sample_sql:
        result = validate_sql(query)
        print("SQL:", query)
        print(result)
        print("-" * 60)
