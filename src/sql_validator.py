"""SQL validation and sanitization utilities for chatbot queries."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, List

try:
    from . import schema_map
except ImportError:  # pragma: no cover
    import schema_map  # type: ignore


@dataclass
class ValidationResult:
    is_valid: bool
    cleaned_sql: str
    error_reason: str | None
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid

    def __iter__(self) -> Iterator[object]:
        yield self.is_valid
        yield self.error_reason or "OK"


_FORBIDDEN_KEYWORDS = ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "EXEC")
_SQL_FUNCTIONS = {"ABS", "AVG", "CAST", "COALESCE", "COUNT", "LOWER", "MAX", "MIN", "ROUND", "SUM", "UPPER"}
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
_NON_COLUMN_TOKENS = {">", "<", "=", ">=", "<=", "!=", "BETWEEN", "AND", "OR", "NOT", "IN", "LIKE", "NULL", "TRUE", "FALSE"}
_VALID_COLUMNS_BY_LOWER = {column.lower(): column for column in schema_map.get_all_column_names()}


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
    cleaned = re.sub(r'\s+AS\s+"[^"]+"\s*$', "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+AS\s+[A-Za-z_][A-Za-z0-9_]*\s*$", "", cleaned, flags=re.IGNORECASE)

    identifiers: List[str] = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?\b", cleaned):
        token = _normalize_identifier(match.group(0))
        upper_token = token.upper()
        if upper_token in _SQL_KEYWORDS or upper_token in _SQL_FUNCTIONS or upper_token in _NON_COLUMN_TOKENS:
            continue
        if token.lower() == "applicants":
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            continue
        identifiers.append(token)
    return identifiers


def check_forbidden_keywords(sql: str) -> str | None:
    text = "" if sql is None else str(sql)
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE):
            return keyword
    return None


def check_table_name(sql: str) -> bool:
    cleaned = sanitize_sql(sql)
    matches = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_\.]*)", cleaned, flags=re.IGNORECASE)
    if not matches:
        return False
    normalized = [_normalize_identifier(name).lower() for name in matches]
    return all(name == "applicants" for name in normalized)


def extract_columns_from_sql(sql: str) -> List[str]:
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

    for clause_pattern in (
        r"\bWHERE\b(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|;|$)",
        r"\bORDER\s+BY\b(.*?)(?:\bLIMIT\b|;|$)",
        r"\bGROUP\s+BY\b(.*?)(?:\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|;|$)",
        r"\bHAVING\b(.*?)(?:\bORDER\s+BY\b|\bLIMIT\b|;|$)",
    ):
        match = re.search(clause_pattern, stripped, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        for expression in _split_top_level_csv(match.group(1)):
            expression = re.sub(r"\bASC\b|\bDESC\b", "", expression, flags=re.IGNORECASE)
            columns.extend(_extract_identifiers(expression))

    return _unique_in_order(columns)


def check_column_names(sql: str) -> List[str]:
    invalid_columns: List[str] = []
    for column_name in extract_columns_from_sql(sql):
        if not column_name or column_name == "*":
            continue
        upper_name = column_name.upper()
        if upper_name in _SQL_FUNCTIONS or upper_name in _SQL_KEYWORDS or upper_name in _NON_COLUMN_TOKENS:
            continue
        if schema_map.is_valid_column(column_name) or column_name.lower() in _VALID_COLUMNS_BY_LOWER:
            continue
        invalid_columns.append(column_name)
    return _unique_in_order(invalid_columns)


def sanitize_sql(sql: str) -> str:
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
    cleaned = sanitize_sql(sql)
    if not cleaned:
        return ValidationResult(False, "", "Empty query", [])
    if len(cleaned) > 2000:
        return ValidationResult(False, cleaned, "Query exceeds maximum allowed length", [])
    if "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        return ValidationResult(False, cleaned, "Comments are not allowed in SQL", [])
    if not re.match(r"^\s*SELECT\b", cleaned, flags=re.IGNORECASE):
        return ValidationResult(False, cleaned, "Only SELECT statements are allowed", [])

    forbidden = check_forbidden_keywords(cleaned)
    if forbidden:
        return ValidationResult(False, cleaned, f"Forbidden SQL keyword detected: {forbidden}", [])
    if ";" in cleaned[:-1]:
        return ValidationResult(False, cleaned, "Only a single SQL statement is allowed", [])
    if not check_table_name(cleaned):
        return ValidationResult(False, cleaned, "Query must reference only the applicants table", [])

    warnings: List[str] = []
    invalid_columns = check_column_names(cleaned)
    if invalid_columns:
        warnings.append("Potentially invalid column names: " + ", ".join(invalid_columns))

    return ValidationResult(True, cleaned, None, warnings)


def is_explanation_query(sql: str) -> bool:
    cleaned = sanitize_sql(sql).lower()
    return "top_features" in cleaned or "explanation_text" in cleaned
