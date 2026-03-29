"""Rule-based query parsing for the chatbot pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List

try:
    from . import schema_map
except ImportError:  # pragma: no cover
    import schema_map  # type: ignore


class IntentType(str, Enum):
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
    intent: str = IntentType.UNKNOWN.value
    applicant_ids: List[int] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    columns_mentioned: List[str] = field(default_factory=list)
    raw_query: str = ""
    is_general: bool = False


_COLUMN_ALIASES = dict(getattr(schema_map, "COLUMN_ALIASES", {}))
_KNOWN_COLUMNS = list(schema_map.get_all_column_names()) if hasattr(schema_map, "get_all_column_names") else []
_COLUMN_BY_LOWER = {column.lower(): column for column in _KNOWN_COLUMNS}
_FIELD_TERMS = sorted({*(_COLUMN_ALIASES.keys()), *(_KNOWN_COLUMNS)}, key=len, reverse=True)
_FIELD_PATTERN = "|".join(re.escape(term) for term in _FIELD_TERMS) or r"(?!x)x"
_NUMERIC_PATTERN = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
_VALUE_PATTERN = rf"(?:{_NUMERIC_PATTERN}|[A-Za-z][\w/-]*)"
_APPLICANT_ID_PATTERN = r"\d{1,12}"

_EXPLAIN_PATTERN = re.compile(r"\b(why|explain|reason|reasons|because)\b", re.IGNORECASE)
_COMPARE_PATTERN = re.compile(r"\b(compare|comparison|vs\.?|versus|difference between)\b", re.IGNORECASE)
_AGGREGATE_PATTERN = re.compile(
    r"\b(average|avg|count|total|sum|min|max|minimum|maximum|how many|what percentage|most common|breakdown|distribution)\b",
    re.IGNORECASE,
)
_CONCEPTUAL_PATTERN = re.compile(r"\b(what is|what are|define|meaning of|tell me about|how does|how do)\b", re.IGNORECASE)
_DECISION_LETTER_PATTERN = re.compile(r"\b(decision letter|generate letter|write letter|formal letter)\b", re.IGNORECASE)
_IMPROVEMENT_PATTERN = re.compile(
    r"(\bwhat would\b.*\b(change|qualify|improve)\b|\bhow can\b.*\b(qualify|get approved|improve)\b|\bwhat do they need\b)",
    re.IGNORECASE,
)
_FILTER_SIGNAL_PATTERN = re.compile(
    r"(>=|<=|!=|=|>|<|\bbetween\b|\bgreater than\b|\bless than\b|\babove\b|\bbelow\b|\bover\b|\bunder\b|\bat least\b|\bat most\b)",
    re.IGNORECASE,
)


def _append_unique(target: List[str], value: str | None) -> None:
    if value and value not in target:
        target.append(value)


def _unique_in_order(values: List[int]) -> List[int]:
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
    original_text = _coerce_query_text(query).strip()
    result = schema_map.fuzzy_resolve(original_text)
    replacements = [
        (r"greater than\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"> \1"),
        (r"more than\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"> \1"),
        (r"above\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"> \1"),
        (r"over\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"> \1"),
        (r"less than\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"< \1"),
        (r"below\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"< \1"),
        (r"under\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"< \1"),
        (r"at least\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r">= \1"),
        (r"at most\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"<= \1"),
        (r"equal to\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"= \1"),
        (r"equals?\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"= \1"),
        (r"between\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)\s+and\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)", r"BETWEEN \1 AND \2"),
    ]
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
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
    if column == "prediction":
        decision_map = {
            "approved": "APPROVE",
            "approve": "APPROVE",
            "accepted": "APPROVE",
            "review": "REVIEW",
            "rejected": "DECLINE",
            "reject": "DECLINE",
            "declined": "DECLINE",
            "decline": "DECLINE",
        }
        return decision_map.get(cleaned, cleaned.upper())
    if column == "risk_band":
        return {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}.get(cleaned, cleaned.upper())
    return cleaned


def _extract_columns_mentioned(query: str) -> List[str]:
    columns: List[str] = []
    for term in _FIELD_TERMS:
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        if re.search(pattern, query, flags=re.IGNORECASE):
            _append_unique(columns, _resolve_column(term))
    return columns


def _extract_columns_from_filters(filters: List[str]) -> List[str]:
    columns: List[str] = []
    for clause in filters:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+(?:BETWEEN|>=|<=|!=|=|>|<)", clause)
        if match:
            _append_unique(columns, match.group(1))
    return columns


def extract_applicant_ids(query: str) -> List[int]:
    text = _coerce_query_text(query).strip()
    if not text:
        return []

    extracted: List[int] = []
    compare_patterns = [
        rf"\bcompare\b.*?\b({_APPLICANT_ID_PATTERN})\b\s*(?:and|vs\.?|versus|with)\s*({_APPLICANT_ID_PATTERN})\b",
        rf"\b({_APPLICANT_ID_PATTERN})\b\s*(?:vs\.?|versus)\s*({_APPLICANT_ID_PATTERN})\b",
    ]
    for pattern in compare_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            extracted.extend([int(match.group(1)), int(match.group(2))])

    for match in re.finditer(
        rf"\b(?:applicant|customer|id|application|application_id|applicant_id|sk_id(?:_curr)?)\b\s*(?:#|:|=)?\s*({_APPLICANT_ID_PATTERN})\b",
        text,
        flags=re.IGNORECASE,
    ):
        extracted.append(int(match.group(1)))

    if not extracted and not _FILTER_SIGNAL_PATTERN.search(text):
        for match in re.finditer(r"\b(\d{4,12})\b", text):
            extracted.append(int(match.group(1)))

    return _unique_in_order(extracted)


def extract_filters(query: str) -> List[str]:
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
            value = _normalize_filter_value(column, match.group("value"))
            _append_unique(filters, f"{column} {match.group('op')} {value}")

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

    for match in re.finditer(r"\b(high|medium|low)\s+risk\b", text, flags=re.IGNORECASE):
        column = _resolve_column("risk")
        value = _normalize_filter_value(column, match.group(1))
        _append_unique(filters, f"{column} = {value}")

    for match in re.finditer(r"\b(approved|review|rejected|declined)\s+applicants?\b", text, flags=re.IGNORECASE):
        column = _resolve_column("decision") or "prediction"
        value = _normalize_filter_value(column, match.group(1))
        _append_unique(filters, f"{column} = {value}")

    return filters


def detect_intent(query: str, ids: List[int], filters: List[str]) -> str:
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
        or re.search(r"\b(applicant|applicants|application|customer|record|records|show|get|find|lookup|list|filter)\b", lowered)
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
    raw_text = _coerce_query_text(raw_query)
    try:
        normalized_text = parse_natural_filter(raw_text)
        filters = extract_filters(normalized_text)
        ids = extract_applicant_ids(normalized_text)
        columns = _extract_columns_mentioned(normalized_text)
        for column in _extract_columns_from_filters(filters):
            _append_unique(columns, column)

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
