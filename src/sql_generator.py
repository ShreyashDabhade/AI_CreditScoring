"""SQLite SQL generation for the credit scoring chatbot.

The primary interface takes a parsed query object and produces a SQL SELECT
statement using Gemini. A small rule-based fallback handles simple lookups and
explanations when Gemini is unavailable or fails.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from gemini_client import get_model

try:
    from . import schema_map
    from .query_parser import IntentType, ParsedQuery, parse_natural_filter
except ImportError:  # pragma: no cover - allows `python src/sql_generator.py`
    import schema_map  # type: ignore
    from query_parser import IntentType, ParsedQuery, parse_natural_filter  # type: ignore

logger = logging.getLogger(__name__)

_REQUEST_OPTIONS = {"timeout": 15}
_SUMMARY_DASHBOARD_COLUMNS = [
    'SK_ID_CURR AS "Applicant ID"',
    'AGE_YEARS AS "Age"',
    'AMT_INCOME_TOTAL_CAPPED AS "Annual Income"',
    'AMT_CREDIT AS "Loan Amount"',
    'AMT_ANNUITY AS "Annual Loan Payment"',
    'CREDIT_INCOME_RATIO AS "Debt-to-Income Ratio"',
    'DAYS_EMPLOYED AS "Days Employed"',
    'EXT_SOURCE_MEAN AS "External Credit Rating"',
    'INST_LATE_COUNT AS "Late Payments Count"',
    'credit_score AS "Internal Credit Score"',
    'risk_band AS "Risk Level"',
    'probability AS "Default Probability"',
    'explanation_text AS "AI Explanation"',
]


def _build_summary_dashboard_select() -> str:
    """Return the curated loan-officer projection used for constrained lists."""

    return ",\n    ".join(_SUMMARY_DASHBOARD_COLUMNS)


def build_sql_prompt(parsed: ParsedQuery) -> str:
    """Build a detailed Gemini prompt for SQL generation."""

    normalized_query = parse_natural_filter(parsed.raw_query)
    resolved_query = schema_map.fuzzy_resolve(normalized_query)
    ids_text = ", ".join(str(applicant_id) for applicant_id in parsed.applicant_ids) or "None"
    filters_text = "\n".join(f"- {filter_clause}" for filter_clause in parsed.filters) or "- None"
    columns_text = ", ".join(parsed.columns_mentioned) if parsed.columns_mentioned else "None"

    prompt = (
        "You are generating safe SQLite SQL for a credit scoring chatbot.\n\n"
        "SCHEMA SECTION:\n"
        f"{schema_map.get_schema_prompt_for_gemini()}\n\n"
        "RULES SECTION:\n"
        "- Table name is always: applicants\n"
        "- Only use SELECT statements, never INSERT/UPDATE/DELETE/DROP\n"
        "- Only use column names from the schema above\n"
        '- For FILTER queries: use this exact projection with aliases: '
        'SK_ID_CURR AS "Applicant ID", AGE_YEARS AS "Age", '
        'AMT_INCOME_TOTAL_CAPPED AS "Annual Income", AMT_CREDIT AS "Loan Amount", '
        'AMT_ANNUITY AS "Annual Loan Payment", CREDIT_INCOME_RATIO AS "Debt-to-Income Ratio", '
        'DAYS_EMPLOYED AS "Days Employed", EXT_SOURCE_MEAN AS "External Credit Rating", '
        'INST_LATE_COUNT AS "Late Payments Count", credit_score AS "Internal Credit Score", '
        'risk_band AS "Risk Level", probability AS "Default Probability", '
        'explanation_text AS "AI Explanation" '
        'FROM applicants WHERE <conditions>\n'
        "- For LOOKUP: SELECT * FROM applicants WHERE SK_ID_CURR = <id>\n"
        "- For COMPARE: SELECT * FROM applicants WHERE SK_ID_CURR IN (<id1>, <id2>)\n"
        "- For EXPLAIN: SELECT explanation_text, top_features, prediction, credit_score, risk_band FROM applicants WHERE SK_ID_CURR = <id>\n"
        "- For AGGREGATE: use COUNT(), AVG(), MIN(), MAX() appropriately\n"
        "- Quote text values with single quotes when building SQL predicates\n"
        "- Always add LIMIT 50 unless user asks for specific count\n"
        "11. If the user query contains a numeric ID after words like SK_ID, applicant, customer, application, id - map it to SK_ID_CURR in the WHERE clause.\n"
        "- Return ONLY the SQL query, no explanation, no markdown\n\n"
        "INTENT SECTION:\n"
        f"- Detected intent: {parsed.intent}\n"
        f"- Applicant IDs: {ids_text}\n"
        f"- Extracted filters:\n{filters_text}\n"
        f"- Columns mentioned: {columns_text}\n"
        f"- General question flag: {parsed.is_general}\n\n"
        "ORIGINAL USER QUERY:\n"
        f"{parsed.raw_query}\n\n"
        "USER QUERY:\n"
        f"{resolved_query}\n\n"
        "OUTPUT INSTRUCTION:\n"
        "Return only the raw SQL query. No markdown. No explanation."
    )
    return prompt


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

    candidates = getattr(response, "candidates", None)
    if candidates:
        parts = []
        for candidate in candidates:
            for attr in ("text", "content", "output"):
                value = getattr(candidate, attr, None)
                if isinstance(value, str) and value.strip():
                    parts.append(value)
            content = getattr(candidate, "content", None)
            if isinstance(content, list):
                for item in content:
                    item_text = getattr(item, "text", None)
                    if isinstance(item_text, str) and item_text.strip():
                        parts.append(item_text)
        if parts:
            return "\n".join(parts)

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
            response = gemini_model.generate_content(
                prompt,
                request_options=_REQUEST_OPTIONS,
            )
        except TypeError:
            try:
                response = gemini_model.generate_content(
                    prompt=prompt,
                    request_options=_REQUEST_OPTIONS,
                )
            except TypeError:
                response = gemini_model.generate_content(prompt=prompt)
        return _extract_response_text(response)

    if hasattr(gemini_model, "invoke"):
        response = gemini_model.invoke(prompt)
        return _extract_response_text(response)

    if hasattr(gemini_model, "generate_text"):
        response = gemini_model.generate_text(prompt=prompt)
        return _extract_response_text(response)

    if callable(gemini_model):
        response = gemini_model(prompt)
        return _extract_response_text(response)

    raise TypeError("Unsupported Gemini model interface")


def _is_numeric_literal(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:,\d{3})*(?:\.\d+)?", value.strip()))


def _quote_sql_literal(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return "''"
    if cleaned.startswith("'") and cleaned.endswith("'"):
        return cleaned
    if _is_numeric_literal(cleaned):
        return cleaned.replace(",", "")
    if cleaned.upper() in {"NULL", "TRUE", "FALSE"}:
        return cleaned.upper()
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
    """Choose the best target column for a deterministic aggregate query."""

    for column_name in parsed.columns_mentioned:
        if column_name != "SK_ID_CURR":
            return column_name
    return None


def _build_aggregate_where_clause(parsed: ParsedQuery) -> str:
    """Render any parsed filters as a SQL WHERE clause for aggregate fallback."""

    if not parsed.filters:
        return ""

    where_parts = [_normalize_filter_clause_for_sql(filter_clause) for filter_clause in parsed.filters]
    return " WHERE " + " AND ".join(where_parts)


def _build_generic_aggregate_sql(parsed: ParsedQuery) -> str | None:
    """Build deterministic MAX/MIN/AVG/COUNT SQL for common aggregate questions."""

    raw_lower = parsed.raw_query.lower()
    target_column = _pick_aggregate_column(parsed)
    where_clause = _build_aggregate_where_clause(parsed)

    if re.search(r"\b(max|maximum|highest|top)\b|max\s*\(", raw_lower):
        if not target_column:
            return None
        return (
            f"SELECT MAX({target_column}) AS max_{target_column.lower()} "
            f"FROM applicants{where_clause};"
        )

    if re.search(r"\b(min|minimum|lowest|smallest)\b|min\s*\(", raw_lower):
        if not target_column:
            return None
        return (
            f"SELECT MIN({target_column}) AS min_{target_column.lower()} "
            f"FROM applicants{where_clause};"
        )

    if re.search(r"\b(avg|average|mean)\b|avg\s*\(", raw_lower):
        if not target_column:
            return None
        return (
            f"SELECT AVG({target_column}) AS avg_{target_column.lower()} "
            f"FROM applicants{where_clause};"
        )

    if re.search(r"\b(count|how many|total number|number of)\b|count\s*\(", raw_lower):
        return f"SELECT COUNT(*) AS count FROM applicants{where_clause};"

    return None


def fallback_sql(parsed: ParsedQuery) -> str | None:
    """Return a simple deterministic SQL query when the case is trivial."""

    if parsed.intent == IntentType.LOOKUP_APPLICANT.value and len(parsed.applicant_ids) == 1:
        applicant_id = int(parsed.applicant_ids[0])
        return f"SELECT * FROM applicants WHERE SK_ID_CURR = {applicant_id} LIMIT 1;"

    if parsed.intent == IntentType.EXPLAIN_APPLICANT.value and len(parsed.applicant_ids) == 1:
        applicant_id = int(parsed.applicant_ids[0])
        return (
            "SELECT SK_ID_CURR, prediction, credit_score, risk_band, probability, "
            "AMT_INCOME_TOTAL_CAPPED, AMT_CREDIT, AGE_YEARS, "
            "top_features, explanation_text "
            f"FROM applicants WHERE SK_ID_CURR = {applicant_id} LIMIT 1;"
        )

    if parsed.intent in {
        IntentType.DECISION_LETTER.value,
        IntentType.IMPROVEMENT_SUGGESTION.value,
    } and len(parsed.applicant_ids) == 1:
        applicant_id = int(parsed.applicant_ids[0])
        return (
            "SELECT SK_ID_CURR, prediction, credit_score, risk_band, probability, "
            "AMT_INCOME_TOTAL_CAPPED, AMT_CREDIT, AGE_YEARS, "
            "top_features, explanation_text "
            f"FROM applicants WHERE SK_ID_CURR = {applicant_id} LIMIT 1;"
        )

    if parsed.intent == IntentType.COMPARE_APPLICANTS.value and len(parsed.applicant_ids) >= 2:
        id_list = ", ".join(str(int(applicant_id)) for applicant_id in parsed.applicant_ids[:2])
        return f"SELECT * FROM applicants WHERE SK_ID_CURR IN ({id_list}) LIMIT 50;"

    if parsed.intent == IntentType.FILTER_APPLICANTS.value and parsed.filters:
        where_parts = [_normalize_filter_clause_for_sql(filter_clause) for filter_clause in parsed.filters]
        where_clause = " AND ".join(where_parts)
        select_clause = _build_summary_dashboard_select()
        return (
            "SELECT\n"
            f"    {select_clause}\n"
            "FROM applicants\n"
            f"WHERE {where_clause}\n"
            "LIMIT 50;"
        )

    if parsed.intent == IntentType.AGGREGATE_QUERY.value:
        raw_lower = parsed.raw_query.lower()
        if ("rejection reason" in raw_lower or "rejected" in raw_lower or "reject" in raw_lower) and "under" in raw_lower:
            age_match = re.search(r"under\s+(\d+)", raw_lower)
            age = age_match.group(1) if age_match else "25"
            return (
                "SELECT top_features, COUNT(*) as freq FROM applicants "
                f"WHERE prediction = 'REJECT' AND AGE_YEARS < {age} "
                "AND top_features IS NOT NULL "
                "GROUP BY top_features ORDER BY freq DESC LIMIT 10;"
            )
        if "rejection reason" in raw_lower or "why rejected" in raw_lower or "top reason" in raw_lower:
            return (
                "SELECT top_features, COUNT(*) as freq FROM applicants "
                "WHERE prediction = 'REJECT' AND top_features IS NOT NULL "
                "GROUP BY top_features ORDER BY freq DESC LIMIT 10;"
            )
        if "how many" in raw_lower and ("reject" in raw_lower or "approv" in raw_lower):
            return (
                "SELECT prediction, COUNT(*) as count "
                "FROM applicants GROUP BY prediction ORDER BY count DESC;"
            )
        if "average income" in raw_lower and "reject" in raw_lower:
            return (
                "SELECT AVG(AMT_INCOME_TOTAL_CAPPED) as avg_income "
                "FROM applicants WHERE prediction = 'REJECT';"
            )
        if "average income" in raw_lower and "approv" in raw_lower:
            return (
                "SELECT AVG(AMT_INCOME_TOTAL_CAPPED) as avg_income "
                "FROM applicants WHERE prediction = 'APPROVE';"
            )
        if "risk" in raw_lower and ("breakdown" in raw_lower or "distribution" in raw_lower or "band" in raw_lower):
            return (
                "SELECT risk_band, prediction, COUNT(*) as count "
                "FROM applicants GROUP BY risk_band, prediction "
                "ORDER BY risk_band, count DESC;"
            )
        if "average" in raw_lower and "score" in raw_lower:
            return (
                "SELECT prediction, AVG(credit_score) as avg_score "
                "FROM applicants GROUP BY prediction;"
            )
        if "average income" in raw_lower:
            return "SELECT AVG(AMT_INCOME_TOTAL_CAPPED) as average_income FROM applicants;"
        if "how many" in raw_lower or "count" in raw_lower or "total" in raw_lower:
            return "SELECT prediction, COUNT(*) as count FROM applicants GROUP BY prediction;"
        if "risk" in raw_lower and ("breakdown" in raw_lower or "distribution" in raw_lower):
            return "SELECT risk_band, COUNT(*) as count FROM applicants GROUP BY risk_band ORDER BY count DESC;"

        generic_aggregate_sql = _build_generic_aggregate_sql(parsed)
        if generic_aggregate_sql:
            return generic_aggregate_sql

    return None


def generate_sql(parsed: ParsedQuery, gemini_model: Any | None) -> str:
    """Generate SQL for a parsed query using Gemini, with safe fallback."""

    prompt = build_sql_prompt(parsed)
    logger.debug("Gemini SQL prompt:\n%s", prompt)

    try:
        model = gemini_model or get_model()

        raw_response = _call_gemini(prompt, model)
        clean_sql = _strip_markdown_fences(raw_response)
        logger.debug("[SQL_GEN] Raw Gemini response: %s", raw_response)
        logger.debug("[SQL_GEN] Cleaned SQL: %s", clean_sql)
        if clean_sql:
            return clean_sql

        fallback = fallback_sql(parsed)
        logger.warning("[SQL_GEN] Empty Gemini SQL response, using fallback: %s", fallback)
        return fallback or ""
    except Exception as exc:
        logger.exception("Gemini SQL generation failed: %s", exc)
        fallback = fallback_sql(parsed)
        logger.warning("[SQL_GEN] Falling back to deterministic SQL: %s", fallback)
        return fallback or ""


if __name__ == "__main__":
    sample = ParsedQuery(
        intent=IntentType.FILTER_APPLICANTS.value,
        applicant_ids=[],
        filters=["income > 50000", "age < 30"],
        columns_mentioned=["income", "age"],
        raw_query="show applicants with income > 50000 and age < 30",
        is_general=False,
    )
    print(build_sql_prompt(sample))
