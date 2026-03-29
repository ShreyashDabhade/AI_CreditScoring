"""SQL generation for the chatbot pipeline."""
from __future__ import annotations

import logging
import re
from typing import Any

from gemini_client import get_model

try:
    from . import schema_map
    from .query_parser import IntentType, ParsedQuery, parse_natural_filter
except ImportError:  # pragma: no cover
    import schema_map  # type: ignore
    from query_parser import IntentType, ParsedQuery, parse_natural_filter  # type: ignore

logger = logging.getLogger(__name__)

_REQUEST_OPTIONS = {"timeout": 15}
_SUMMARY_DASHBOARD_COLUMNS = [
    'application_id AS "Application ID"',
    'SK_ID_CURR AS "Applicant ID"',
    'applicant_name AS "Applicant Name"',
    'AGE_YEARS AS "Age"',
    'AMT_INCOME_TOTAL_CAPPED AS "Annual Income"',
    'AMT_CREDIT AS "Loan Amount"',
    'AMT_ANNUITY AS "Annual Loan Payment"',
    'CREDIT_INCOME_RATIO AS "Debt-to-Income Ratio"',
    'EXT_SOURCE_MEAN AS "External Credit Rating"',
    'INST_LATE_COUNT AS "Late Payments Count"',
    'credit_score AS "Internal Credit Score"',
    'risk_band AS "Risk Level"',
    'probability AS "Default Probability"',
    'prediction AS "Decision"',
    'explanation_text AS "AI Explanation"',
]


def _build_summary_dashboard_select() -> str:
    return ",\n    ".join(_SUMMARY_DASHBOARD_COLUMNS)


def build_sql_prompt(parsed: ParsedQuery) -> str:
    normalized_query = parse_natural_filter(parsed.raw_query)
    resolved_query = schema_map.fuzzy_resolve(normalized_query)
    ids_text = ", ".join(str(applicant_id) for applicant_id in parsed.applicant_ids) or "None"
    filters_text = "\n".join(f"- {filter_clause}" for filter_clause in parsed.filters) or "- None"
    columns_text = ", ".join(parsed.columns_mentioned) if parsed.columns_mentioned else "None"

    return (
        "You are generating safe SQLite SQL for a credit scoring chatbot.\n\n"
        "SCHEMA SECTION:\n"
        f"{schema_map.get_schema_prompt_for_gemini()}\n\n"
        "RULES SECTION:\n"
        "- Table name is always: applicants\n"
        "- Only use SELECT statements\n"
        "- Only use column names from the schema above\n"
        "- For FILTER queries: use the exact projection shown in the prompt if possible\n"
        "- For LOOKUP: select the applicant row using SK_ID_CURR or application_id\n"
        "- For COMPARE: select the two matching applicants\n"
        "- For EXPLAIN: include explanation_text, top_features, prediction, credit_score, risk_band, probability\n"
        "- For AGGREGATE: use COUNT(), AVG(), MIN(), MAX() appropriately\n"
        "- Quote text values with single quotes\n"
        "- Always add LIMIT 50 unless user asked for a specific count\n"
        "- Return ONLY raw SQL\n\n"
        "INTENT SECTION:\n"
        f"- Detected intent: {parsed.intent}\n"
        f"- Applicant IDs: {ids_text}\n"
        f"- Extracted filters:\n{filters_text}\n"
        f"- Columns mentioned: {columns_text}\n\n"
        "ORIGINAL USER QUERY:\n"
        f"{parsed.raw_query}\n\n"
        "USER QUERY:\n"
        f"{resolved_query}\n"
    )


def _strip_markdown_fences(text: str) -> str:
    cleaned = "" if text is None else str(text).strip()
    cleaned = re.sub(r"^\s*sql\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    for attr in ("text", "content", "output"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(response, dict):
        for key in ("text", "content", "output"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value
    if isinstance(response, str):
        return response
    return str(response).strip()


def _call_gemini(prompt: str, gemini_model: Any) -> str:
    if gemini_model is None:
        raise RuntimeError("Gemini model is not available")
    if hasattr(gemini_model, "generate_content"):
        try:
            response = gemini_model.generate_content(prompt, request_options=_REQUEST_OPTIONS)
        except TypeError:
            response = gemini_model.generate_content(prompt=prompt)
        return _extract_response_text(response)
    if callable(gemini_model):
        return _extract_response_text(gemini_model(prompt))
    raise TypeError("Unsupported Gemini model interface")


def _is_numeric_literal(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:,\d{3})*(?:\.\d+)?", value.strip()))


def _quote_sql_literal(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("'") and cleaned.endswith("'"):
        return cleaned
    if _is_numeric_literal(cleaned):
        return cleaned.replace(",", "")
    return "'" + cleaned.replace("'", "''") + "'"


def _normalize_filter_clause_for_sql(filter_clause: str) -> str:
    between_match = re.match(
        r"^(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s+BETWEEN\s+(?P<low>.+?)\s+AND\s+(?P<high>.+)$",
        filter_clause,
        flags=re.IGNORECASE,
    )
    if between_match:
        return (
            f"{between_match.group('column')} BETWEEN "
            f"{_quote_sql_literal(between_match.group('low'))} AND "
            f"{_quote_sql_literal(between_match.group('high'))}"
        )

    comparison_match = re.match(
        r"^(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<op>>=|<=|!=|=|>|<)\s*(?P<value>.+)$",
        filter_clause,
    )
    if comparison_match:
        return (
            f"{comparison_match.group('column')} {comparison_match.group('op')} "
            f"{_quote_sql_literal(comparison_match.group('value'))}"
        )

    return filter_clause


def _pick_aggregate_column(parsed: ParsedQuery) -> str | None:
    for column_name in parsed.columns_mentioned:
        if column_name not in {"SK_ID_CURR", "application_id", "applicant_id"}:
            return column_name
    return None


def _build_aggregate_where_clause(parsed: ParsedQuery) -> str:
    if not parsed.filters:
        return ""
    where_parts = [_normalize_filter_clause_for_sql(filter_clause) for filter_clause in parsed.filters]
    return " WHERE " + " AND ".join(where_parts)


def _build_generic_aggregate_sql(parsed: ParsedQuery) -> str | None:
    raw_lower = parsed.raw_query.lower()
    target_column = _pick_aggregate_column(parsed)
    where_clause = _build_aggregate_where_clause(parsed)

    if re.search(r"\b(max|maximum|highest|top)\b", raw_lower):
        return f"SELECT MAX({target_column}) AS max_value FROM applicants{where_clause};" if target_column else None
    if re.search(r"\b(min|minimum|lowest|smallest)\b", raw_lower):
        return f"SELECT MIN({target_column}) AS min_value FROM applicants{where_clause};" if target_column else None
    if re.search(r"\b(avg|average|mean)\b", raw_lower):
        return f"SELECT AVG({target_column}) AS avg_value FROM applicants{where_clause};" if target_column else None
    if re.search(r"\b(count|how many|total number|number of)\b", raw_lower):
        return f"SELECT COUNT(*) AS count FROM applicants{where_clause};"
    return None


def fallback_sql(parsed: ParsedQuery) -> str | None:
    if parsed.intent == IntentType.LOOKUP_APPLICANT.value and len(parsed.applicant_ids) == 1:
        applicant_id = int(parsed.applicant_ids[0])
        return (
            "SELECT * FROM applicants "
            f"WHERE SK_ID_CURR = {applicant_id} OR application_id = {applicant_id} LIMIT 1;"
        )

    if parsed.intent in {
        IntentType.EXPLAIN_APPLICANT.value,
        IntentType.DECISION_LETTER.value,
        IntentType.IMPROVEMENT_SUGGESTION.value,
    } and len(parsed.applicant_ids) == 1:
        applicant_id = int(parsed.applicant_ids[0])
        return (
            "SELECT application_id, SK_ID_CURR, applicant_name, prediction, credit_score, risk_band, probability, "
            "AMT_INCOME_TOTAL_CAPPED, AMT_CREDIT, AGE_YEARS, top_features, explanation_text "
            "FROM applicants "
            f"WHERE SK_ID_CURR = {applicant_id} OR application_id = {applicant_id} LIMIT 1;"
        )

    if parsed.intent == IntentType.COMPARE_APPLICANTS.value and len(parsed.applicant_ids) >= 2:
        id_list = ", ".join(str(int(applicant_id)) for applicant_id in parsed.applicant_ids[:2])
        return (
            "SELECT * FROM applicants "
            f"WHERE SK_ID_CURR IN ({id_list}) OR application_id IN ({id_list}) LIMIT 50;"
        )

    if parsed.intent == IntentType.FILTER_APPLICANTS.value and parsed.filters:
        where_parts = [_normalize_filter_clause_for_sql(filter_clause) for filter_clause in parsed.filters]
        select_clause = _build_summary_dashboard_select()
        return (
            "SELECT\n"
            f"    {select_clause}\n"
            "FROM applicants\n"
            f"WHERE {' AND '.join(where_parts)}\n"
            "LIMIT 50;"
        )

    if parsed.intent == IntentType.AGGREGATE_QUERY.value:
        raw_lower = parsed.raw_query.lower()
        if "rejection reason" in raw_lower or "decline reason" in raw_lower or "why rejected" in raw_lower or "why declined" in raw_lower:
            return (
                "SELECT top_features, COUNT(*) as count "
                "FROM applicants WHERE prediction = 'DECLINE' AND top_features IS NOT NULL "
                "GROUP BY top_features ORDER BY count DESC LIMIT 10;"
            )
        if "how many" in raw_lower and any(token in raw_lower for token in ("declin", "reject", "approv", "review")):
            return "SELECT prediction, COUNT(*) as count FROM applicants GROUP BY prediction ORDER BY count DESC;"
        if "average income" in raw_lower and any(token in raw_lower for token in ("declin", "reject")):
            return "SELECT AVG(AMT_INCOME_TOTAL_CAPPED) as avg_income FROM applicants WHERE prediction = 'DECLINE';"
        if "average income" in raw_lower and "approv" in raw_lower:
            return "SELECT AVG(AMT_INCOME_TOTAL_CAPPED) as avg_income FROM applicants WHERE prediction = 'APPROVE';"
        if "average income" in raw_lower:
            return "SELECT AVG(AMT_INCOME_TOTAL_CAPPED) as average_income FROM applicants;"
        generic = _build_generic_aggregate_sql(parsed)
        if generic:
            return generic

    return None


def generate_sql(parsed: ParsedQuery, gemini_model: Any | None) -> str:
    prompt = build_sql_prompt(parsed)
    try:
        model = gemini_model or get_model()
        raw_response = _call_gemini(prompt, model)
        clean_sql = _strip_markdown_fences(raw_response)
        if clean_sql:
            return clean_sql
    except Exception as exc:
        logger.debug("Gemini SQL generation failed: %s", exc)
    return fallback_sql(parsed) or ""
