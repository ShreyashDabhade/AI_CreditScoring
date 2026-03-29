"""Natural-language response generation for the chatbot pipeline."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from gemini_client import get_model

try:
    from .query_parser import IntentType, ParsedQuery
    from .sql_executor import ExecutionResult, format_results_as_text
except ImportError:  # pragma: no cover
    from query_parser import IntentType, ParsedQuery  # type: ignore
    from sql_executor import ExecutionResult, format_results_as_text  # type: ignore

logger = logging.getLogger(__name__)
_REQUEST_OPTIONS = {"timeout": 15}


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _truncate_value(value: Any, max_length: int = 120) -> str:
    text = _safe_json(value) if isinstance(value, (dict, list, tuple)) else ("" if value is None else str(value))
    return text if len(text) <= max_length else text[: max_length - 3] + "..."


def _format_row(row: dict) -> str:
    return "- " + "; ".join(f"{key}: {_truncate_value(value)}" for key, value in row.items()) if row else "- <empty row>"


def _normalize_decision_value(decision: Any) -> str:
    text = "Unknown" if decision is None else str(decision).strip()
    lowered = text.lower()
    if text == "1" or lowered in {"reject", "rejected", "decline", "declined"}:
        return "DECLINE"
    if text == "0" or lowered in {"approve", "approved"}:
        return "APPROVE"
    if lowered == "review":
        return "REVIEW"
    return text.upper() if lowered in {"approve", "approved", "review", "decline", "declined"} else text


def _probability_percent(probability: Any) -> str:
    try:
        return f"{float(probability) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _decision_past_tense(decision: Any) -> str:
    normalized = _normalize_decision_value(decision)
    if normalized == "APPROVE":
        return "approved"
    if normalized == "DECLINE":
        return "declined"
    if normalized == "REVIEW":
        return "sent to review"
    return str(normalized).lower()


def _extract_reason_phrases(top_features: Any, limit: int = 3) -> list[str]:
    phrases: list[str] = []
    if isinstance(top_features, str):
        text = top_features.strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                top_features = json.loads(text)
            except Exception:
                top_features = text
    if isinstance(top_features, str):
        phrases = [part.strip() for part in re.split(r"[;,]", top_features) if part.strip()]
    elif isinstance(top_features, (list, tuple)):
        for item in top_features:
            if isinstance(item, dict):
                feature_name = item.get("feature") or item.get("name")
                reason = item.get("reason")
                if feature_name and reason:
                    phrases.append(f"{feature_name} ({reason})")
                elif feature_name:
                    phrases.append(str(feature_name).strip())
            elif item:
                phrases.append(str(item).strip())
    elif isinstance(top_features, dict):
        value = top_features.get("feature") or top_features.get("name")
        if value:
            phrases.append(str(value).strip())
    return phrases[:limit]


def _action_for_reason(reason: str) -> str:
    lowered = reason.lower()
    if "debt" in lowered or "income ratio" in lowered:
        return "reducing existing debt or increasing stable income before reapplying"
    if "employment" in lowered or "job" in lowered:
        return "building a longer and more stable employment record"
    if "late payment" in lowered or "missed" in lowered or "delin" in lowered:
        return "showing a consistent pattern of on-time repayments"
    if "income" in lowered:
        return "strengthening verifiable income or reducing the requested loan amount"
    if "credit" in lowered or "loan" in lowered:
        return "applying for a smaller loan amount that is easier to support"
    return "addressing the main affordability and repayment concerns in the application"


def _human_join(items: list[str]) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _format_metric_value(value: Any) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def _numeric_summary(rows: list[dict]) -> str:
    numeric_ranges: list[str] = []
    for column in sorted({key for row in rows for key in row.keys()}):
        numeric_values: list[float] = []
        for row in rows:
            value = row.get(column)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
        if numeric_values:
            numeric_ranges.append(f"{column}: min={min(numeric_values):g}, max={max(numeric_values):g}")
    return "\n".join(f"- {item}" for item in numeric_ranges) if numeric_ranges else "No numeric range summary available."


def _build_data_section(result: ExecutionResult) -> str:
    if not result.success:
        return f"Execution status: failed\nFallback summary:\n{format_results_as_text(result)}"
    if result.row_count == 0:
        return "Execution status: succeeded\nData note: the query returned 0 rows."
    if result.row_count <= 5:
        return "Execution status: succeeded\nReturned rows:\n" + "\n".join(_format_row(row) for row in result.rows)
    return (
        "Execution status: succeeded\n"
        f"Returned rows: {result.row_count}\n"
        "First 5 rows:\n"
        f"{chr(10).join(_format_row(row) for row in result.rows[:5])}\n"
        "Summary stats:\n"
        f"- Count: {result.row_count}\n"
        f"{_numeric_summary(result.rows)}"
    )


def build_response_prompt(parsed: ParsedQuery, result: ExecutionResult, raw_query: str) -> str:
    aggregate_instruction = (
        "- Summarize these aggregate statistics in 2-3 plain English sentences. Include the actual numbers.\n"
        if parsed.intent == IntentType.AGGREGATE_QUERY.value
        else ""
    )
    return (
        "ROLE:\n"
        "You are a credit risk analyst assistant explaining applicant data to a business user.\n\n"
        "CONTEXT:\n"
        f"- Original question: {raw_query}\n"
        f"- Detected intent: {parsed.intent}\n"
        f"- Query succeeded: {result.success}\n"
        f"- Number of results returned: {result.row_count}\n\n"
        "DATA:\n"
        f"{_build_data_section(result)}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer the user's question directly using the data\n"
        "- For EXPLAIN_APPLICANT: explain the decision in plain English using explanation_text and top_features\n"
        "- For FILTER and AGGREGATE: summarize patterns found\n"
        "- For COMPARE: highlight key differences between applicants\n"
        f"{aggregate_instruction}"
        "- Keep response under 200 words\n"
        "- Do not mention SQL or database internals\n"
    )


def build_explain_prompt(row: dict[str, Any]) -> str:
    applicant_id = row.get("SK_ID_CURR", row.get("applicant_id", row.get("application_id", "Unknown")))
    decision = _normalize_decision_value(row.get("prediction", "Unknown"))
    return (
        "Explain this loan decision in plain English to a business user.\n"
        f"Applicant ID: {applicant_id}\n"
        f"Decision: {decision}\n"
        f"Credit Score: {row.get('credit_score', 'N/A')}\n"
        f"Risk Category: {row.get('risk_band', 'N/A')}\n"
        f"Default Probability: {_probability_percent(row.get('probability', 0))}\n"
        f"Income: {row.get('AMT_INCOME_TOTAL_CAPPED', 'N/A')}\n"
        f"Loan Requested: {row.get('AMT_CREDIT', 'N/A')}\n"
        f"Age: {row.get('AGE_YEARS', 'N/A')}\n"
        f"Top factors: {_safe_json(row.get('top_features', 'Not available'))}\n"
        f"Explanation text: {row.get('explanation_text', 'Not available')}\n"
    )


def build_explain_fallback(row: dict[str, Any]) -> str:
    applicant_id = row.get("SK_ID_CURR", row.get("applicant_id", row.get("application_id", "Unknown")))
    decision = _normalize_decision_value(row.get("prediction", "Unknown"))
    risk_band = row.get("risk_band", "N/A")
    probability_pct = _probability_percent(row.get("probability", 0))
    explanation = str(row.get("explanation_text", "") or "").strip()
    reasons = _extract_reason_phrases(row.get("top_features", "Not available"))

    if decision == "APPROVE":
        opening = (
            f"Applicant {applicant_id} was approved and the case was assessed as "
            f"{str(risk_band).lower()} risk with an estimated default probability of {probability_pct}."
        )
        practical = "In practical terms, the overall profile appears strong enough to support the requested borrowing."
    elif decision == "DECLINE":
        opening = (
            f"Applicant {applicant_id} was declined because the case was assessed as "
            f"{str(risk_band).lower()} risk with an estimated default probability of {probability_pct}."
        )
        practical = "In practical terms, the current profile does not yet look strong enough to support the requested borrowing safely."
    else:
        opening = (
            f"Applicant {applicant_id} was sent to review after being assessed as "
            f"{str(risk_band).lower()} risk with an estimated default probability of {probability_pct}."
        )
        practical = "In practical terms, the case is close enough to policy boundaries that it needs analyst review before a final decision."

    reason_sentence = f"The main factors behind that outcome were {_human_join(reasons[:3])}." if reasons else ""
    context_sentence = explanation if explanation and explanation.lower() not in {"not available", decision.lower()} else ""
    return " ".join(part for part in [opening, reason_sentence, context_sentence, practical] if part)


def build_decision_letter_fallback(row: dict[str, Any]) -> str:
    applicant_id = row.get("SK_ID_CURR", row.get("applicant_id", row.get("application_id", "Unknown")))
    decision = _normalize_decision_value(row.get("prediction", "Unknown"))
    credit_score = row.get("credit_score", "N/A")
    risk_band = row.get("risk_band", "N/A")
    probability_pct = _probability_percent(row.get("probability", 0))
    loan_amount = row.get("AMT_CREDIT", "N/A")
    reasons = _extract_reason_phrases(row.get("top_features", "Not available"))
    explanation = str(row.get("explanation_text", "") or "").strip()

    if reasons:
        reason_sentence = f"This decision was mainly influenced by {_human_join(reasons[:3])}."
    elif explanation:
        reason_sentence = explanation if explanation.endswith(".") else f"{explanation}."
    else:
        reason_sentence = f"The application was assessed as {str(risk_band).lower()} risk with an estimated default risk of {probability_pct}."

    if decision == "APPROVE":
        body = (
            f"We are pleased to inform you that your loan application has been approved. "
            f"{reason_sentence} The decision reflects a score of {credit_score} and supports the requested loan amount of {loan_amount}, subject to final verification and standard lending terms."
        )
    else:
        next_step = _action_for_reason(reasons[0] if reasons else explanation)
        body = (
            f"After reviewing your application, we regret to inform you that it has not been approved at this time. "
            f"{reason_sentence} To strengthen a future application, we recommend {next_step}."
        )

    return f"Dear Applicant (Ref: {applicant_id}),\n\n{body}\n\nYours sincerely,\nCredit Risk Department"


def build_improvement_fallback(row: dict[str, Any]) -> str:
    applicant_id = row.get("SK_ID_CURR", row.get("applicant_id", row.get("application_id", "Unknown")))
    decision = _normalize_decision_value(row.get("prediction", "Unknown"))
    credit_score = row.get("credit_score", "N/A")
    reasons = _extract_reason_phrases(row.get("top_features", "Not available"))
    if decision == "APPROVE":
        return f"Applicant {applicant_id} was already approved with a score of {credit_score}. No improvement is needed before qualifying."
    actions: list[str] = []
    for reason in reasons:
        action = _action_for_reason(reason)
        if action not in actions:
            actions.append(action)
    first_action = actions[0] if actions else "strengthening affordability and repayment stability before reapplying"
    second_action = actions[1] if len(actions) > 1 else "showing a stronger and more consistent credit profile over time"
    return (
        f"Applicant {applicant_id} was not approved this time. "
        f"The strongest improvement would be {first_action}. "
        f"It would also help to focus on {second_action}. "
        "With progress in those areas, the applicant should be in a better position to reapply."
    )


def build_compare_fallback(rows: list[dict[str, Any]]) -> str:
    if len(rows) < 2:
        return "Comparison requires exactly two applicants."
    a, b = rows[0], rows[1]
    a_id = a.get("SK_ID_CURR", a.get("application_id", "Unknown"))
    b_id = b.get("SK_ID_CURR", b.get("application_id", "Unknown"))
    a_reasons = _extract_reason_phrases(a.get("top_features", "Not available"))
    b_reasons = _extract_reason_phrases(b.get("top_features", "Not available"))
    separating_factor = b_reasons[0] if b_reasons else (a_reasons[0] if a_reasons else "overall repayment profile")
    return (
        f"Applicant {a_id} was {_decision_past_tense(a.get('prediction', 'Unknown'))}, while applicant {b_id} was {_decision_past_tense(b.get('prediction', 'Unknown'))}. "
        f"The clearest difference is in overall risk: applicant {a_id} has a score of {a.get('credit_score', 'N/A')}, {str(a.get('risk_band', 'N/A')).lower()} risk, and estimated default risk of {_probability_percent(a.get('probability', 0))}, "
        f"compared with {b.get('credit_score', 'N/A')}, {str(b.get('risk_band', 'N/A')).lower()} risk, and {_probability_percent(b.get('probability', 0))} for applicant {b_id}. "
        f"The factor that most separates them is {separating_factor}."
    )


def build_aggregate_fallback(result: ExecutionResult) -> str:
    if not result.success:
        return format_results_as_text(result)
    if not result.rows:
        return "No aggregate data was found for that query."
    if len(result.rows) == 1:
        row = result.rows[0]
        if "average_income" in row:
            return f"The average income across the matching applicants is {_format_metric_value(row.get('average_income', 'N/A'))}."
        if "avg_income" in row:
            return f"The average income across the matching applicants is {_format_metric_value(row.get('avg_income', 'N/A'))}."
        if "count" in row and len(row) == 1:
            return f"The total count for this query is {_format_metric_value(row.get('count', 'N/A'))}."
        details = [f"{key.replace('_', ' ')} is {_format_metric_value(value)}" for key, value in row.items()]
        return f"The aggregate result shows that {_human_join(details)}."
    if all(isinstance(row, dict) and "count" in row for row in result.rows):
        label_key = next((key for key in result.rows[0].keys() if key.lower() != "count"), None)
        if label_key:
            breakdown: list[str] = []
            for row in result.rows:
                label_value = row.get(label_key, "Unknown")
                if label_key.lower() == "prediction":
                    label_text = f"{_decision_past_tense(label_value)} applicants"
                elif label_key.lower() == "risk_band":
                    label_text = f"{str(label_value).lower()}-risk applicants"
                else:
                    label_text = str(label_value)
                breakdown.append(f"{_format_metric_value(row.get('count', 'N/A'))} {label_text}")
            return f"The result shows {_human_join(breakdown)}."
    return f"The query returned {result.row_count} aggregate rows. The first result was {_safe_json(result.rows[0])}."


def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    for attr in ("text", "content", "output"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(response, dict):
        for key in ("text", "content", "output"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(response, str):
        return response.strip()
    return str(response).strip()


def _call_gemini(prompt: str, gemini_model: Any) -> str:
    if gemini_model is None:
        raise RuntimeError("Response model unavailable")
    if hasattr(gemini_model, "generate_content"):
        try:
            response = gemini_model.generate_content(prompt, request_options=_REQUEST_OPTIONS)
        except TypeError:
            response = gemini_model.generate_content(prompt=prompt)
        return _extract_response_text(response)
    if callable(gemini_model):
        return _extract_response_text(gemini_model(prompt))
    raise TypeError("Unsupported Gemini model interface")


def generate_response(parsed: ParsedQuery, result: ExecutionResult, raw_query: str, gemini_model: Any) -> str:
    try:
        active_model = gemini_model
        if active_model is None and parsed.intent != IntentType.AGGREGATE_QUERY.value:
            try:
                active_model = get_model()
            except Exception:
                active_model = None

        if parsed.intent == IntentType.DECISION_LETTER.value:
            return build_decision_letter_fallback(result.rows[0]) if result.success and result.rows else "No data found for the requested applicant."
        if parsed.intent == IntentType.IMPROVEMENT_SUGGESTION.value:
            return build_improvement_fallback(result.rows[0]) if result.success and result.rows else "No data found for the requested applicant."
        if parsed.intent == IntentType.COMPARE_APPLICANTS.value:
            return build_compare_fallback(result.rows) if result.success else "Comparison requires two applicants."
        if parsed.intent == IntentType.EXPLAIN_APPLICANT.value:
            if result.success and result.rows:
                if active_model is not None:
                    try:
                        text = _call_gemini(build_explain_prompt(result.rows[0]), active_model)
                        if text:
                            return text
                    except Exception as exc:
                        logger.debug("Gemini explain failed: %s", exc)
                return build_explain_fallback(result.rows[0])
            return "No data found for the requested applicant."
        if parsed.intent == IntentType.GENERAL_QUESTION.value:
            return handle_general_question(raw_query, active_model)
        if active_model is None:
            return build_aggregate_fallback(result) if parsed.intent == IntentType.AGGREGATE_QUERY.value else format_results_as_text(result)

        prompt = build_response_prompt(parsed, result, raw_query)
        try:
            response_text = _call_gemini(prompt, active_model)
            if response_text:
                return response_text
        except Exception:
            logger.exception("Failed to generate natural-language response")
        return build_aggregate_fallback(result) if parsed.intent == IntentType.AGGREGATE_QUERY.value else format_results_as_text(result)
    except Exception:
        logger.exception("Failed to generate natural-language response")
        return format_results_as_text(result)


def handle_general_question(query: str, gemini_model: Any) -> str:
    prompt = f"You are a credit analyst. Answer this question about credit scoring concisely: {query}"
    active_model = gemini_model
    if active_model is None:
        try:
            active_model = get_model()
        except Exception:
            active_model = None
    if active_model is None:
        return "I can help with credit scoring questions, but I could not generate a concise answer right now."
    try:
        response_text = _call_gemini(prompt, active_model)
        if response_text:
            return response_text
    except Exception:
        logger.exception("Failed to answer general credit question")
    return "I can help with credit scoring questions, but I could not generate a concise answer right now."
