"""Rule-based query parsing for the credit scoring chatbot.

This module converts raw user text into a structured representation that can
be used before SQL generation. It stays intentionally lightweight and depends
only on the local ``schema_map`` module for alias resolution.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Sequence

try:
    from . import schema_map
except ImportError:  # pragma: no cover - allows `python src/query_parser.py`
    import schema_map  # type: ignore


class IntentType(str, Enum):
    """Supported top-level user intents."""

    LOOKUP_APPLICANT = "LOOKUP_APPLICANT"
    FILTER_APPLICANTS = "FILTER_APPLICANTS"
    COMPARE_APPLICANTS = "COMPARE_APPLICANTS"
    EXPLAIN_APPLICANT = "EXPLAIN_APPLICANT"
    DECISION_LETTER = "DECISION_LETTER"
    IMPROVEMENT_SUGGESTION = "IMPROVEMENT_SUGGESTION"
    AGGREGATE_QUERY = "AGGREGATE_QUERY"
    GENERAL_QUESTION = "GENERAL_QUESTION"
    UNKNOWN = "UNKNOWN"


@dataclass
class ParsedQuery:
    """Structured representation of a parsed user query."""

    intent: str = IntentType.UNKNOWN.value
    applicant_ids: List[int] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    columns_mentioned: List[str] = field(default_factory=list)
    raw_query: str = ""
    is_general: bool = False


_COLUMN_ALIASES: Dict[str, str] = dict(getattr(schema_map, "COLUMN_ALIASES", {}))
_KNOWN_COLUMNS: List[str] = (
    list(schema_map.get_all_column_names()) if hasattr(schema_map, "get_all_column_names") else []
)
_COLUMN_BY_LOWER: Dict[str, str] = {column.lower(): column for column in _KNOWN_COLUMNS}
_PREFERRED_ALIAS_BY_COLUMN: Dict[str, str] = {
    "SK_ID_CURR": "applicant_id",
    "AMT_INCOME_TOTAL_CAPPED": "income",
    "AGE_YEARS": "age",
    "applicant_id": "applicant_id",
    "income": "income",
    "age": "age",
    "credit_score": "score",
    "risk_band": "risk",
    "prediction": "decision",
}

_FIELD_TERMS: List[str] = sorted(
    {*(term for term in _COLUMN_ALIASES.keys()), *(_KNOWN_COLUMNS)},
    key=len,
    reverse=True,
)
_FIELD_PATTERN = "|".join(re.escape(term) for term in _FIELD_TERMS) or r"(?!x)x"
_NUMERIC_PATTERN = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
_VALUE_PATTERN = rf"(?:{_NUMERIC_PATTERN}|[A-Za-z][\w/-]*)"

_EXPLAIN_PATTERN = re.compile(r"\b(why|explain|reason|reasons|because)\b", re.IGNORECASE)
_COMPARE_PATTERN = re.compile(r"\b(compare|comparison|vs\.?|versus|difference between)\b", re.IGNORECASE)
_AGGREGATE_PATTERN = re.compile(
    r"\b("
    r"average|avg|count|total|sum|min|max|minimum|maximum|how many|"
    r"what percentage|top rejection reason|most common|count of|breakdown of|"
    r"distribution of|what is the average|total number"
    r")\b",
    re.IGNORECASE,
)
_CONCEPTUAL_PATTERN = re.compile(
    r"\b(what is|what are|define|definition|meaning of|tell me about|how does|how do|explain)\b",
    re.IGNORECASE,
)
_DECISION_LETTER_PATTERN = re.compile(
    r"\b(decision letter|generate letter|write letter|formal letter|rejection letter)\b",
    re.IGNORECASE,
)
_IMPROVEMENT_PATTERN = re.compile(
    r"("
    r"\bwhat would\b.*\b(change|qualify|improve)\b|"
    r"\bhow can\b.*\b(qualify|get approved)\b|"
    r"\bwhat do they need\b"
    r")",
    re.IGNORECASE,
)
_FILTER_SIGNAL_PATTERN = re.compile(
    r"(>=|<=|!=|=|>|<|\bbetween\b|\bgreater than\b|\bless than\b|\babove\b|\bbelow\b|\bover\b|\bunder\b|\bat least\b|\bat most\b)",
    re.IGNORECASE,
)


def _append_unique(target: List[str], value: str | None) -> None:
    if value and value not in target:
        target.append(value)


def _unique_in_order(values: Iterable[int]) -> List[int]:
    seen: set[int] = set()
    ordered: List[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _coerce_query_text(query: str | None) -> str:
    return "" if query is None else str(query)


def parse_natural_filter(query: str) -> str:
    """Convert natural-language comparison phrases into SQL-style operators."""
    original_text = _coerce_query_text(query).strip()
    protected_text = original_text
    protected_tokens = {
        "__VAL0__": "approved",
        "__VAL1__": "rejected",
        "__VAL2__": "declined",
    }
    protected_text = re.sub(r"\bapproved(?=\s+applicants?\b)", "__VAL0__", protected_text, flags=re.IGNORECASE)
    protected_text = re.sub(r"\brejected(?=\s+applicants?\b)", "__VAL1__", protected_text, flags=re.IGNORECASE)
    protected_text = re.sub(r"\bdeclined(?=\s+applicants?\b)", "__VAL2__", protected_text, flags=re.IGNORECASE)

    result = schema_map.fuzzy_resolve(protected_text)
    if not result:
        return result

    replacements = [
        (r"greater than\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"> \1"),
        (r"more than\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"> \1"),
        (r"higher than\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"> \1"),
        (r"above\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"> \1"),
        (r"over\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"> \1"),
        (r"exceeds?\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"> \1"),
        (r"less than\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"< \1"),
        (r"lower than\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"< \1"),
        (r"below\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"< \1"),
        (r"under\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"< \1"),
        (r"at least\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r">= \1"),
        (r"minimum of\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r">= \1"),
        (r"at most\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"<= \1"),
        (r"maximum of\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"<= \1"),
        (r"equals?\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"= \1"),
        (r"equal to\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"= \1"),
        (r"is\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"= \1"),
        (
            r"between\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)\s+and\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)",
            r"BETWEEN \1 AND \2",
        ),
    ]

    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    for placeholder, original_value in protected_tokens.items():
        result = result.replace(placeholder, original_value)

    return result


def _resolve_column(term: str) -> str | None:
    cleaned = term.strip().strip("'\"")
    if not cleaned:
        return None

    resolved = schema_map.resolve_alias(cleaned)
    if resolved:
        return resolved

    return _COLUMN_BY_LOWER.get(cleaned.lower())


def _clean_value_token(value: str) -> str:
    cleaned = value.strip().strip("'\"").rstrip("?.!,;:")
    if re.fullmatch(_NUMERIC_PATTERN, cleaned):
        return cleaned.replace(",", "")
    return cleaned.lower()


def _normalize_filter_value(column: str | None, value: str) -> str:
    cleaned = _clean_value_token(value)

    if not column:
        return cleaned

    if column == "prediction":
        decision_map = {
            "approved": "0",
            "approve": "0",
            "accepted": "0",
            "rejected": "1",
            "reject": "1",
            "declined": "1",
            "decline": "1",
            "default": "1",
        }
        return decision_map.get(cleaned, cleaned)

    if column == "risk_band":
        band_map = {
            "low": "LOW",
            "medium": "MEDIUM",
            "high": "HIGH",
        }
        return band_map.get(cleaned, cleaned.upper())

    return cleaned


def _extract_columns_mentioned(query: str) -> List[str]:
    columns: List[str] = []
    for term in _FIELD_TERMS:
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        if re.search(pattern, query, flags=re.IGNORECASE):
            _append_unique(columns, _resolve_column(term))
    return columns


def _extract_columns_from_filters(filters: Sequence[str]) -> List[str]:
    columns: List[str] = []
    for clause in filters:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+(?:BETWEEN|>=|<=|!=|=|>|<)", clause)
        if match:
            _append_unique(columns, match.group(1))
    return columns


def extract_applicant_ids(query: str) -> List[int]:
    """Extract applicant identifiers from text.

    The parser prefers applicant-specific contexts so numeric filter values
    like ``income > 50000`` are not misclassified as applicant IDs.
    """

    text = _coerce_query_text(query).strip()
    if not text:
        return []

    extracted: List[int] = []

    compare_patterns = [
        r"\bcompare\b.*?\b(\d{1,6})\b\s*(?:and|vs\.?|versus|with)\s*(\d{1,6})\b",
        r"\b(\d{1,6})\b\s*(?:vs\.?|versus)\s*(\d{1,6})\b",
    ]
    for pattern in compare_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            extracted.extend([int(match.group(1)), int(match.group(2))])

    for match in re.finditer(
        r"\b(?:applicant|customer|id|applicant_id|sk_id(?:_curr)?|sk\s*id(?:\s*curr)?)\b\s*(?:#|:|=)?\s*(\d{1,6})\b",
        text,
        flags=re.IGNORECASE,
    ):
        extracted.append(int(match.group(1)))

    if not extracted and not _FILTER_SIGNAL_PATTERN.search(text):
        for match in re.finditer(r"\b(\d{4,6})\b", text):
            extracted.append(int(match.group(1)))

    return _unique_in_order(extracted)


def extract_filters(query: str) -> List[str]:
    """Extract normalized filter clauses from natural language."""

    text = _coerce_query_text(query).strip()
    if not text:
        return []

    filters: List[str] = []

    between_pattern = re.compile(
        rf"\b(?P<field>{_FIELD_PATTERN})\b\s+between\s+(?P<low>{_NUMERIC_PATTERN})\s+and\s+(?P<high>{_NUMERIC_PATTERN})\b",
        re.IGNORECASE,
    )
    for match in between_pattern.finditer(text):
        column = _resolve_column(match.group("field"))
        if column:
            low = _clean_value_token(match.group("low"))
            high = _clean_value_token(match.group("high"))
            _append_unique(filters, f"{column} BETWEEN {low} AND {high}")

    symbolic_pattern = re.compile(
        rf"\b(?P<field>{_FIELD_PATTERN})\b\s*(?P<op>>=|<=|!=|=|>|<)\s*(?P<value>{_VALUE_PATTERN})\b",
        re.IGNORECASE,
    )
    for match in symbolic_pattern.finditer(text):
        column = _resolve_column(match.group("field"))
        if column:
            operator = match.group("op")
            value = _normalize_filter_value(column, match.group("value"))
            _append_unique(filters, f"{column} {operator} {value}")

    word_operator_patterns = [
        (r"greater than|more than|above|over", ">"),
        (r"less than|below|under", "<"),
        (r"at least|minimum of|minimum|no less than", ">="),
        (r"at most|maximum of|maximum|no more than", "<="),
        (r"equal to|equals", "="),
    ]
    for phrase_pattern, operator in word_operator_patterns:
        regex = re.compile(
            rf"\b(?P<field>{_FIELD_PATTERN})\b\s+(?:is\s+)?(?:{phrase_pattern})\s+(?P<value>{_VALUE_PATTERN})\b",
            re.IGNORECASE,
        )
        for match in regex.finditer(text):
            column = _resolve_column(match.group("field"))
            if column:
                value = _normalize_filter_value(column, match.group("value"))
                _append_unique(filters, f"{column} {operator} {value}")

    equality_pattern = re.compile(
        rf"\b(?P<field>{_FIELD_PATTERN})\b\s+is\s+(?P<value>[A-Za-z][\w/-]*)\b",
        re.IGNORECASE,
    )
    for match in equality_pattern.finditer(text):
        column = _resolve_column(match.group("field"))
        if column:
            value = _normalize_filter_value(column, match.group("value"))
            _append_unique(filters, f"{column} = {value}")

    for match in re.finditer(r"\b(high|medium|low)\s+risk\b", text, flags=re.IGNORECASE):
        column = _resolve_column("risk")
        value = _normalize_filter_value(column, match.group(1))
        _append_unique(filters, f"{column} = {value}")

    for match in re.finditer(r"\b(approved|rejected|declined)\s+applicants?\b", text, flags=re.IGNORECASE):
        column = _resolve_column("decision") or _resolve_column("prediction")
        value = _normalize_filter_value(column, match.group(1))
        _append_unique(filters, f"{column} = {value}")

    return filters


def detect_intent(query: str, ids: List[int], filters: List[str]) -> str:
    """Detect the dominant query intent using simple rules."""

    text = _coerce_query_text(query).strip()
    lowered = text.lower()

    if not lowered:
        return IntentType.UNKNOWN.value

    has_decision_letter = bool(_DECISION_LETTER_PATTERN.search(lowered))
    has_improvement = bool(_IMPROVEMENT_PATTERN.search(lowered))
    has_explain = bool(_EXPLAIN_PATTERN.search(lowered))
    has_compare = bool(_COMPARE_PATTERN.search(lowered))
    has_aggregate = bool(_AGGREGATE_PATTERN.search(lowered))
    has_conceptual_signal = bool(_CONCEPTUAL_PATTERN.search(lowered))
    has_db_signal = bool(
        ids
        or filters
        or has_compare
        or has_aggregate
        or re.search(r"\b(applicant|applicants|customer|record|records|show|get|find|lookup|list|filter)\b", lowered)
    )

    if ids and has_decision_letter:
        return IntentType.DECISION_LETTER.value
    if ids and has_improvement:
        return IntentType.IMPROVEMENT_SUGGESTION.value
    if ids and has_explain:
        return IntentType.EXPLAIN_APPLICANT.value
    if ids and has_compare:
        return IntentType.COMPARE_APPLICANTS.value
    if ids:
        return IntentType.LOOKUP_APPLICANT.value
    if has_aggregate:
        return IntentType.AGGREGATE_QUERY.value
    if filters:
        return IntentType.FILTER_APPLICANTS.value
    if has_conceptual_signal or not has_db_signal:
        return IntentType.GENERAL_QUESTION.value
    return IntentType.UNKNOWN.value


def parse_query(raw_query: str) -> ParsedQuery:
    """Parse a raw natural language query into structured fields."""

    raw_text = _coerce_query_text(raw_query)

    try:
        normalized_text = parse_natural_filter(raw_text)
        filters = extract_filters(normalized_text)
        ids = extract_applicant_ids(normalized_text)

        columns = _extract_columns_mentioned(normalized_text)
        for column in _extract_columns_from_filters(filters):
            _append_unique(columns, column)
        if ids:
            _append_unique(columns, _resolve_column("applicant id") or "applicant_id")

        intent = detect_intent(raw_text, ids, filters)
        return ParsedQuery(
            intent=intent,
            applicant_ids=ids,
            filters=filters,
            columns_mentioned=columns,
            raw_query=raw_text,
            is_general=intent == IntentType.GENERAL_QUESTION.value,
        )
    except Exception:
        return ParsedQuery(raw_query=raw_text)


def _coerce_legacy_value(value: str) -> Any:
    cleaned = _clean_value_token(value)
    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)
    if re.fullmatch(r"-?\d+\.\d+", cleaned):
        return float(cleaned)
    return cleaned


def _legacy_filter_map(filter_clauses: Sequence[str]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}

    for clause in filter_clauses:
        between_match = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)\s+BETWEEN\s+(.+?)\s+AND\s+(.+)$",
            clause,
            flags=re.IGNORECASE,
        )
        if between_match:
            column = between_match.group(1)
            entry: Dict[str, Any] = {
                "between": [
                    _coerce_legacy_value(between_match.group(2)),
                    _coerce_legacy_value(between_match.group(3)),
                ]
            }
            filters[column] = entry
            alias = _PREFERRED_ALIAS_BY_COLUMN.get(column)
            if alias:
                filters[alias] = entry
            continue

        comparison_match = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|!=|=|>|<)\s*(.+)$",
            clause,
        )
        if not comparison_match:
            continue

        column = comparison_match.group(1)
        operator = comparison_match.group(2)
        value = _coerce_legacy_value(comparison_match.group(3))
        entry = {operator: value}
        filters[column] = entry
        alias = _PREFERRED_ALIAS_BY_COLUMN.get(column)
        if alias:
            filters[alias] = entry

    return filters


def _map_to_legacy_intent(parsed: ParsedQuery) -> str:
    if parsed.intent == IntentType.LOOKUP_APPLICANT.value:
        return "GET_APPLICANT"
    if parsed.intent == IntentType.COMPARE_APPLICANTS.value:
        return "COMPARISON_QUERY"
    if parsed.intent in {
        IntentType.EXPLAIN_APPLICANT.value,
        IntentType.DECISION_LETTER.value,
        IntentType.IMPROVEMENT_SUGGESTION.value,
    }:
        return "EXPLANATION_QUERY"
    if parsed.intent == IntentType.FILTER_APPLICANTS.value:
        if any(clause.startswith(("risk_band ", "prediction ")) for clause in parsed.filters):
            return "RISK_QUERY"
        return "FILTER_QUERY"
    if parsed.intent == IntentType.AGGREGATE_QUERY.value:
        return "FILTER_QUERY"
    return "GENERAL_QUERY"


def parse_intent(user_query: str) -> Dict[str, Any]:
    """Backward-compatible wrapper around :func:`parse_query`.

    Existing routes still consume the old dictionary shape, so this adapter
    keeps them stable while the richer parser is available to new code.
    """

    parsed = parse_query(user_query)
    return {
        "intent": _map_to_legacy_intent(parsed),
        "applicant_ids": parsed.applicant_ids,
        "filters": _legacy_filter_map(parsed.filters),
        "filter_clauses": parsed.filters,
        "columns_mentioned": parsed.columns_mentioned,
        "needs_explanation": parsed.intent == IntentType.EXPLAIN_APPLICANT.value,
        "is_comparison": parsed.intent == IntentType.COMPARE_APPLICANTS.value,
        "raw_query": parsed.raw_query,
        "parsed_query": parsed,
    }


if __name__ == "__main__":
    sample_queries = [
        "show me applicant 12345",
        "show applicants with income > 50000 and age < 30",
        "compare applicant 100 and 200",
        "why was applicant 99999 rejected?",
        "what is a credit score?",
    ]

    for query in sample_queries:
        print(f"Query: {query}")
        print(asdict(parse_query(query)))
        print("-" * 60)
