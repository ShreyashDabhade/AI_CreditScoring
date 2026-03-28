"""Natural-language response generation for the credit scoring chatbot."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from gemini_client import get_model

try:
    from .query_parser import IntentType, ParsedQuery
    from .sql_executor import ExecutionResult, format_results_as_text
except ImportError:  # pragma: no cover - allows `python src/response_generator.py`
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
    if isinstance(value, (dict, list, tuple)):
        text = _safe_json(value)
    else:
        text = "" if value is None else str(value)
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def _format_row(row: dict) -> str:
    if not row:
        return "- <empty row>"
    parts = [f"{key}: {_truncate_value(value)}" for key, value in row.items()]
    return "- " + "; ".join(parts)


def _normalize_decision_value(decision: Any) -> str:
    text = "Unknown" if decision is None else str(decision).strip()
    if text == "1":
        return "REJECT"
    if text == "0":
        return "APPROVE"
    return text.upper() if text.lower() in {"approve", "approved", "reject", "rejected"} else text


def _probability_percent(probability: Any) -> str:
    try:
        value = float(probability)
    except (TypeError, ValueError):
        value = 0.0
    return f"{value * 100:.1f}%"


def _decision_past_tense(decision: Any) -> str:
    """Return a natural-language past-tense decision label."""
    normalized = _normalize_decision_value(decision)
    if normalized == "APPROVE":
        return "approved"
    if normalized == "REJECT":
        return "rejected"
    return str(normalized).lower()


def _extract_reason_phrases(top_features: Any, limit: int = 3) -> list[str]:
    phrases: list[str] = []

    if isinstance(top_features, str):
        phrases = [part.strip() for part in re.split(r"[;,]", top_features) if part.strip()]
    elif isinstance(top_features, (list, tuple)):
        for item in top_features:
            if isinstance(item, dict):
                feature_name = item.get("feature") or item.get("name")
                if feature_name:
                    phrases.append(str(feature_name).strip())
            elif item:
                phrases.append(str(item).strip())
    elif isinstance(top_features, dict):
        for key in ("feature", "name"):
            value = top_features.get(key)
            if value:
                phrases.append(str(value).strip())

    return phrases[:limit]


def _action_for_reason(reason: str) -> str:
    lowered = reason.lower()
    if "debt" in lowered or "income ratio" in lowered:
        return "reducing existing debt or increasing stable income before reapplying"
    if "employment" in lowered or "job" in lowered:
        return "building a longer and more stable employment record"
    if "late payment" in lowered or "delin" in lowered:
        return "showing a consistent pattern of on-time repayments"
    if "income" in lowered:
        return "strengthening verifiable income or reducing the requested loan amount"
    if "loan" in lowered or "credit" in lowered:
        return "applying for a smaller loan amount that is easier to support"
    return "addressing the main affordability and repayment concerns in the application"


def _human_join(items: list[str]) -> str:
    """Join short phrases into a natural-language list."""
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _format_metric_value(value: Any) -> str:
    """Format a scalar metric for user-facing prose."""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _numeric_summary(rows: list[dict]) -> str:
    numeric_ranges: list[str] = []
    if not rows:
        return "No rows available for summary."

    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())

    for column in sorted(all_columns):
        numeric_values: list[float] = []
        for row in rows:
            value = row.get(column)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))

        if numeric_values:
            minimum = min(numeric_values)
            maximum = max(numeric_values)
            numeric_ranges.append(f"{column}: min={minimum:g}, max={maximum:g}")

    if not numeric_ranges:
        return "No numeric range summary available."

    return "\n".join(f"- {item}" for item in numeric_ranges)


def _build_data_section(result: ExecutionResult) -> str:
    if not result.success:
        return (
            "Execution status: failed\n"
            "Data note: no reliable result rows are available.\n"
            "Fallback summary:\n"
            f"{format_results_as_text(result)}"
        )

    if result.row_count == 0:
        return "Execution status: succeeded\nData note: the query returned 0 rows."

    if result.row_count <= 5:
        rows_text = "\n".join(_format_row(row) for row in result.rows)
        return (
            "Execution status: succeeded\n"
            f"Returned rows ({result.row_count}):\n"
            f"{rows_text}"
        )

    preview_rows = "\n".join(_format_row(row) for row in result.rows[:5])
    summary_text = _numeric_summary(result.rows)
    return (
        "Execution status: succeeded\n"
        f"Returned rows: {result.row_count}\n"
        "First 5 rows:\n"
        f"{preview_rows}\n"
        "Summary stats:\n"
        f"- Count: {result.row_count}\n"
        f"{summary_text}"
    )


def build_response_prompt(
    parsed: ParsedQuery,
    result: ExecutionResult,
    raw_query: str,
) -> str:
    """Build a Gemini prompt for the final user-facing response."""
    aggregate_instruction = ""
    if parsed.intent == IntentType.AGGREGATE_QUERY.value:
        aggregate_instruction = (
            "- Summarize these aggregate statistics in 2-3 plain English sentences. "
            "Include the actual numbers. Do not use bullet points.\n"
        )

    prompt = (
        "ROLE:\n"
        "You are a credit risk analyst assistant explaining loan decisions and applicant data to a business user.\n\n"
        "CONTEXT:\n"
        f"- Original question: {raw_query}\n"
        f"- Detected intent: {parsed.intent}\n"
        f"- Query succeeded: {result.success}\n"
        f"- Number of results returned: {result.row_count}\n\n"
        "DATA:\n"
        f"{_build_data_section(result)}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer the user's question directly using the data\n"
        "- For EXPLAIN_APPLICANT: explain the decision in plain English using explanation_text and top_features columns\n"
        "- For FILTER/AGGREGATE: summarize patterns found\n"
        "- For GENERAL_QUESTION: answer from credit domain knowledge (no data needed)\n"
        "- For COMPARE: highlight key differences between applicants\n"
        f"{aggregate_instruction}"
        "- Keep response under 200 words unless explanation is requested\n"
        "- Do NOT mention SQL, database, or technical internals\n"
        "- Be professional but friendly"
    )
    return prompt


def build_explain_prompt(row: dict[str, Any]) -> str:
    """Build a rich Gemini prompt for applicant-level decision explanations."""
    applicant_id = row.get("SK_ID_CURR", row.get("applicant_id", "Unknown"))
    decision = _normalize_decision_value(row.get("prediction", "Unknown"))
    credit_score = row.get("credit_score", "N/A")
    risk_band = row.get("risk_band", "N/A")
    probability = row.get("probability", 0)
    income = row.get("AMT_INCOME_TOTAL_CAPPED", row.get("income", "N/A"))
    loan_amount = row.get("AMT_CREDIT", row.get("mock_credit", "N/A"))
    age = row.get("AGE_YEARS", row.get("age", "N/A"))
    days_employed = row.get("DAYS_EMPLOYED", "N/A")
    top_features = row.get("top_features", "Not available")
    explanation = row.get("explanation_text", "Not available")

    probability_pct = _probability_percent(probability)

    try:
        employment_years: int | str = abs(int(float(days_employed))) // 365
    except (TypeError, ValueError):
        employment_years = "N/A"

    if isinstance(top_features, (dict, list, tuple)):
        top_features = _safe_json(top_features)
    if explanation is None:
        explanation = "Not available"

    return (
        "You are a senior credit risk analyst explaining a loan decision \n"
        "to a business user (loan officer, not a developer).\n\n"
        "STRICT RULES:\n"
        "- Do NOT mention SQL, databases, column names, or any technical terms\n"
        "- Do NOT use bullet points or lists\n"
        "- Write in clear, flowing paragraphs (3-4 sentences total)\n"
        "- Be direct and factual but warm and professional\n"
        "- Explain the WHY, not just the WHAT\n\n"
        "APPLICANT FACTS:\n"
        f"Applicant ID    : {applicant_id}\n"
        f"Decision        : {decision}\n"
        f"Credit Score    : {credit_score} / 850\n"
        f"Risk Category   : {risk_band}\n"
        f"Default Probability: {probability_pct}\n"
        f"Annual Income   : {income}\n"
        f"Loan Requested  : {loan_amount}\n"
        f"Age             : {age} years\n"
        f"Employment      : {employment_years} years\n\n"
        "KEY RISK DRIVERS (SHAP analysis - most important factors):\n"
        f"{top_features}\n\n"
        "SYSTEM NARRATIVE:\n"
        f"{explanation}\n\n"
        "YOUR TASK:\n"
        "Write 3-4 sentences that explain:\n"
        "1. What the final decision was and how confident the model is \n"
        "   (use the default probability to phrase this naturally)\n"
        "2. The 2-3 most important reasons from the SHAP analysis above, \n"
        "   explained in plain English without jargon\n"
        "3. What this means practically for this applicant\n\n"
        "Do not start with \"Based on\" or \"According to\". \n"
        "Start directly with the applicant's situation.\n"
    )


def build_decision_letter_prompt(row: dict[str, Any]) -> str:
    """Generate a formal credit decision letter prompt for Gemini."""
    applicant_id = row.get("SK_ID_CURR", row.get("applicant_id", "Unknown"))
    decision = _normalize_decision_value(row.get("prediction", "Unknown"))
    credit_score = row.get("credit_score", "N/A")
    risk_band = row.get("risk_band", "N/A")
    probability = row.get("probability", 0)
    income = row.get("AMT_INCOME_TOTAL_CAPPED", row.get("income", "N/A"))
    loan_amount = row.get("AMT_CREDIT", row.get("mock_credit", "N/A"))
    top_features = row.get("top_features", "Not available")
    explanation = row.get("explanation_text", "Not available")
    prob_pct = _probability_percent(probability)

    if isinstance(top_features, (dict, list, tuple)):
        top_features = _safe_json(top_features)

    return (
        "You are a credit officer at a financial institution writing a formal\n"
        "loan decision letter to send to an applicant.\n\n"
        "Write a professional, formal letter that:\n"
        f'1. Opens with "Dear Applicant (Ref: {applicant_id}),"\n'
        "2. States the decision clearly in the first paragraph\n"
        "3. Cites 2-3 specific reasons from the SHAP factors below in plain English\n"
        "   (no jargon, no column names, no technical terms)\n"
        "4. If REJECTED: includes one sentence about what the applicant could\n"
        "   improve to strengthen a future application\n"
        "5. If APPROVED: includes loan terms summary\n"
        '6. Closes formally with "Yours sincerely, Credit Risk Department"\n\n'
        "Keep the letter under 200 words. Professional tone throughout.\n\n"
        "APPLICANT DATA:\n"
        f"Reference     : {applicant_id}\n"
        f"Decision      : {decision}\n"
        f"Credit Score  : {credit_score}/850\n"
        f"Risk Category : {risk_band}\n"
        f"Default Risk  : {prob_pct}\n"
        f"Income        : {income}\n"
        f"Loan Requested: {loan_amount}\n\n"
        "KEY FACTORS (use these to explain the decision):\n"
        f"{top_features}\n\n"
        "SYSTEM NARRATIVE:\n"
        f"{explanation}\n\n"
        "Write the letter now:\n"
    )


def build_improvement_prompt(row: dict[str, Any]) -> str:
    """Build a prompt describing how a rejected applicant could qualify later."""
    applicant_id = row.get("SK_ID_CURR", row.get("applicant_id", "Unknown"))
    decision = _normalize_decision_value(row.get("prediction", "Unknown"))
    credit_score = row.get("credit_score", "N/A")
    risk_band = row.get("risk_band", "N/A")
    top_features = row.get("top_features", "Not available")
    probability = row.get("probability", 0)
    prob_pct = _probability_percent(probability)

    if isinstance(top_features, (dict, list, tuple)):
        top_features = _safe_json(top_features)

    if decision == "APPROVE":
        return (
            f"Applicant {applicant_id} was already approved with a credit score of "
            f"{credit_score}. No improvement needed."
        )

    return (
        "You are a credit counsellor helping a rejected loan applicant understand\n"
        "what they need to improve to qualify in the future.\n\n"
        "APPLICANT DATA:\n"
        f"ID            : {applicant_id}\n"
        f"Decision      : {decision}\n"
        f"Credit Score  : {credit_score}/850\n"
        f"Risk Level    : {risk_band}\n"
        f"Default Risk  : {prob_pct}\n\n"
        "KEY NEGATIVE FACTORS (SHAP analysis):\n"
        f"{top_features}\n\n"
        "YOUR TASK:\n"
        "Write 3-4 sentences in plain English that:\n"
        "1. Acknowledge the rejection empathetically (one sentence)\n"
        "2. Identify the top 2 specific things they should improve based on\n"
        "   the SHAP factors above - translate each factor into actionable\n"
        "   advice (e.g. \"high debt-to-income ratio\" -> \"reducing existing debt\n"
        "   or increasing income before reapplying\")\n"
        "3. End with an encouraging sentence about reapplying\n\n"
        "Do not use column names, technical terms, or jargon.\n"
        "Be warm, honest, and constructive.\n"
    )


def build_compare_prompt(rows: list[dict[str, Any]]) -> str:
    """Build a Gemini prompt for side-by-side applicant comparison."""
    if len(rows) < 2:
        return "Comparison requires exactly two applicants."

    a, b = rows[0], rows[1]
    top_factors_a = a.get("top_features", "N/A")
    top_factors_b = b.get("top_features", "N/A")
    if isinstance(top_factors_a, (dict, list, tuple)):
        top_factors_a = _safe_json(top_factors_a)
    if isinstance(top_factors_b, (dict, list, tuple)):
        top_factors_b = _safe_json(top_factors_b)

    def fmt(row: dict[str, Any]) -> str:
        return (
            f"  ID: {row.get('SK_ID_CURR', row.get('applicant_id', 'Unknown'))} | "
            f"Decision: {_normalize_decision_value(row.get('prediction', 'Unknown'))} | "
            f"Score: {row.get('credit_score', 'N/A')} | "
            f"Risk: {row.get('risk_band', 'N/A')} | "
            f"Default prob: {_probability_percent(row.get('probability', 0))} | "
            f"Income: {row.get('AMT_INCOME_TOTAL_CAPPED', row.get('income', 'N/A'))} | "
            f"Loan: {row.get('AMT_CREDIT', row.get('mock_credit', 'N/A'))} | "
            f"Age: {row.get('AGE_YEARS', row.get('age', 'N/A'))}"
        )

    return (
        "You are a credit analyst comparing two loan applicants side by side.\n\n"
        "APPLICANT A:\n"
        f"{fmt(a)}\n"
        f"Top factors: {top_factors_a}\n\n"
        "APPLICANT B:\n"
        f"{fmt(b)}\n"
        f"Top factors: {top_factors_b}\n\n"
        "YOUR TASK:\n"
        "Write 3-4 sentences that:\n"
        "1. State both decisions upfront\n"
        "2. Highlight the 2-3 most significant differences between them\n"
        "   that explain why their outcomes differ (or are similar)\n"
        "3. Identify which factor most separates them\n\n"
        "Plain English only. No column names. No bullet points. Be analytical.\n"
    )


def build_explain_fallback(row: dict[str, Any]) -> str:
    """Build a deterministic applicant explanation when Gemini is unavailable."""
    applicant_id = row.get("SK_ID_CURR", row.get("applicant_id", "Unknown"))
    decision = _normalize_decision_value(row.get("prediction", "Unknown"))
    risk_band = row.get("risk_band", "N/A")
    probability_pct = _probability_percent(row.get("probability", 0))
    explanation = str(row.get("explanation_text", "") or "").strip()
    reasons = _extract_reason_phrases(row.get("top_features", "Not available"))

    if decision == "APPROVE":
        opening = (
            f"Applicant {applicant_id} was approved, and the application was assessed as "
            f"{str(risk_band).lower()} risk with an estimated default probability of {probability_pct}."
        )
        practical = "In practical terms, the overall profile appears strong enough to support the requested borrowing."
    else:
        opening = (
            f"Applicant {applicant_id} was rejected because the application was assessed as "
            f"{str(risk_band).lower()} risk, with an estimated default probability of {probability_pct}."
        )
        practical = "In practical terms, the current profile does not yet look strong enough to support the requested borrowing safely."

    reason_sentence = ""
    if reasons:
        reason_sentence = f"The main factors behind that decision were {_human_join(reasons[:3])}."
    elif explanation and explanation.lower() != "not available":
        reason_sentence = explanation if explanation.endswith(".") else f"{explanation}."

    context_sentence = ""
    if explanation and explanation.lower() != "not available" and explanation not in reason_sentence:
        context_sentence = explanation if explanation.endswith(".") else f"{explanation}."

    parts = [opening, reason_sentence, context_sentence, practical]
    return " ".join(part for part in parts if part)


def build_decision_letter_fallback(row: dict[str, Any]) -> str:
    """Build a formal decision letter without relying on Gemini."""
    applicant_id = row.get("SK_ID_CURR", row.get("applicant_id", "Unknown"))
    decision = _normalize_decision_value(row.get("prediction", "Unknown"))
    credit_score = row.get("credit_score", "N/A")
    risk_band = row.get("risk_band", "N/A")
    probability_pct = _probability_percent(row.get("probability", 0))
    loan_amount = row.get("AMT_CREDIT", row.get("mock_credit", "N/A"))
    explanation = str(row.get("explanation_text", "") or "").strip()
    reasons = _extract_reason_phrases(row.get("top_features", "Not available"))

    reason_sentence = ""
    if reasons:
        reason_sentence = f"This decision was mainly influenced by {_human_join(reasons[:3])}."
    elif explanation and explanation.lower() != "not available":
        reason_sentence = explanation if explanation.endswith(".") else f"{explanation}."
    else:
        reason_sentence = (
            f"The application was assessed as {str(risk_band).lower()} risk with an estimated default risk of {probability_pct}."
        )

    if decision == "APPROVE":
        body = (
            f"We are pleased to inform you that your loan application has been approved. "
            f"{reason_sentence} The decision reflects a credit score of {credit_score} and supports the requested loan amount of {loan_amount}, subject to final verification and standard lending terms."
        )
    else:
        next_step = _action_for_reason(reasons[0] if reasons else explanation)
        body = (
            f"After reviewing your application, we regret to inform you that it has not been approved at this time. "
            f"{reason_sentence} To strengthen a future application, we recommend {next_step}."
        )

    return (
        f"Dear Applicant (Ref: {applicant_id}),\n\n"
        f"{body}\n\n"
        "Yours sincerely,\n"
        "Credit Risk Department"
    )


def build_improvement_fallback(row: dict[str, Any]) -> str:
    """Build constructive qualification guidance without Gemini."""
    applicant_id = row.get("SK_ID_CURR", row.get("applicant_id", "Unknown"))
    decision = _normalize_decision_value(row.get("prediction", "Unknown"))
    credit_score = row.get("credit_score", "N/A")
    reasons = _extract_reason_phrases(row.get("top_features", "Not available"))

    if decision == "APPROVE":
        return (
            f"Applicant {applicant_id} was already approved with a credit score of {credit_score}. "
            "No improvement is needed before qualifying."
        )

    actions: list[str] = []
    for reason in reasons:
        action = _action_for_reason(reason)
        if action not in actions:
            actions.append(action)

    first_action = actions[0] if actions else "strengthening affordability and repayment stability before reapplying"
    second_action = actions[1] if len(actions) > 1 else "showing a stronger and more consistent credit profile over time"

    return (
        f"Applicant {applicant_id} was not approved this time, and that can be frustrating. "
        f"The strongest improvement would be {first_action}. "
        f"It would also help to focus on {second_action}. "
        "With progress in those areas, the applicant should be in a better position to reapply."
    )


def build_compare_fallback(rows: list[dict[str, Any]]) -> str:
    """Build a deterministic comparison narrative for two applicants."""
    if len(rows) < 2:
        return "Comparison requires exactly two applicants."

    a, b = rows[0], rows[1]
    a_id = a.get("SK_ID_CURR", a.get("applicant_id", "Unknown"))
    b_id = b.get("SK_ID_CURR", b.get("applicant_id", "Unknown"))
    a_decision = _decision_past_tense(a.get("prediction", "Unknown"))
    b_decision = _decision_past_tense(b.get("prediction", "Unknown"))
    a_score = a.get("credit_score", "N/A")
    b_score = b.get("credit_score", "N/A")
    a_risk = str(a.get("risk_band", "N/A")).lower()
    b_risk = str(b.get("risk_band", "N/A")).lower()
    a_prob = _probability_percent(a.get("probability", 0))
    b_prob = _probability_percent(b.get("probability", 0))
    a_income = a.get("AMT_INCOME_TOTAL_CAPPED", a.get("income", "N/A"))
    b_income = b.get("AMT_INCOME_TOTAL_CAPPED", b.get("income", "N/A"))
    a_loan = a.get("AMT_CREDIT", a.get("mock_credit", "N/A"))
    b_loan = b.get("AMT_CREDIT", b.get("mock_credit", "N/A"))
    a_reasons = _extract_reason_phrases(a.get("top_features", "Not available"))
    b_reasons = _extract_reason_phrases(b.get("top_features", "Not available"))

    separating_factor = b_reasons[0] if b_reasons else (a_reasons[0] if a_reasons else "overall repayment profile")

    return (
        f"Applicant {a_id} was {a_decision}, while applicant {b_id} was {b_decision}. "
        f"The clearest difference is in overall risk: applicant {a_id} has a credit score of {a_score}, {a_risk} risk, and an estimated default risk of {a_prob}, compared with {b_score}, {b_risk} risk, and {b_prob} for applicant {b_id}. "
        f"The income and borrowing picture also differs, with applicant {a_id} at {a_income} income and a requested loan of {a_loan}, versus {b_income} income and {b_loan} for applicant {b_id}. "
        f"The factor that most separates them is {separating_factor}."
    )


def build_aggregate_fallback(result: ExecutionResult) -> str:
    """Summarize aggregate query results in plain English without Gemini."""
    if not result.success:
        return format_results_as_text(result)

    if not result.rows:
        return "No aggregate data was found for that query."

    if len(result.rows) == 1:
        row = result.rows[0]
        if "average_income" in row:
            return (
                "The average income across the matching applicants is "
                f"{_format_metric_value(row.get('average_income', 'N/A'))}."
            )

        if "count" in row and len(row) == 1:
            return f"The total count for this query is {_format_metric_value(row.get('count', 'N/A'))}."

        details = [
            f"{key.replace('_', ' ')} is {_format_metric_value(value)}"
            for key, value in row.items()
        ]
        return f"The aggregate result shows that {_human_join(details)}."

    if all(isinstance(row, dict) and "count" in row for row in result.rows):
        label_key = next(
            (
                key
                for key in result.rows[0].keys()
                if key.lower() != "count"
            ),
            None,
        )
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

    return (
        f"The query returned {result.row_count} aggregate rows. "
        f"The first result was {_safe_json(result.rows[0])}."
    )


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

    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            text = _extract_response_text(candidate)
            if text:
                return text

    if isinstance(response, str):
        return response.strip()

    return str(response).strip()


def _call_gemini(prompt: str, gemini_model: Any) -> str:
    if gemini_model is None:
        raise RuntimeError("Response model unavailable")

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
        return _extract_response_text(gemini_model.invoke(prompt))

    if hasattr(gemini_model, "generate_text"):
        try:
            response = gemini_model.generate_text(prompt=prompt)
        except TypeError:
            response = gemini_model.generate_text(prompt)
        return _extract_response_text(response)

    if callable(gemini_model):
        return _extract_response_text(gemini_model(prompt))

    raise TypeError("Unsupported Gemini model interface")


def generate_response(
    parsed: ParsedQuery,
    result: ExecutionResult,
    raw_query: str,
    gemini_model: Any,
) -> str:
    """Generate a user-facing response or fall back to a safe text summary."""

    try:
        active_model = gemini_model
        if active_model is None and parsed.intent != IntentType.AGGREGATE_QUERY.value:
            try:
                active_model = get_model()
            except Exception:
                active_model = None

        if parsed.intent == IntentType.DECISION_LETTER.value:
            if result.success and result.rows:
                prompt = build_decision_letter_prompt(result.rows[0])
                try:
                    response_text = _call_gemini(prompt, active_model)
                    if response_text:
                        return response_text
                except Exception as exc:
                    logger.error("[LETTER] Gemini failed: %s", exc)
                return build_decision_letter_fallback(result.rows[0])
            return "No data found for the requested applicant."

        if parsed.intent == IntentType.IMPROVEMENT_SUGGESTION.value:
            if result.success and result.rows:
                prompt = build_improvement_prompt(result.rows[0])
                if "YOUR TASK:" not in prompt:
                    return prompt
                try:
                    response_text = _call_gemini(prompt, active_model)
                    if response_text:
                        return response_text
                except Exception as exc:
                    logger.error("[IMPROVEMENT] Gemini failed: %s", exc)
                return build_improvement_fallback(result.rows[0])
            return "No data found for the requested applicant."

        if parsed.intent == IntentType.COMPARE_APPLICANTS.value:
            if result.success and len(result.rows) >= 2:
                prompt = build_compare_prompt(result.rows)
                try:
                    response_text = _call_gemini(prompt, active_model)
                    if response_text:
                        return response_text
                except Exception as exc:
                    logger.error("[COMPARE] Gemini failed: %s", exc)
                return build_compare_fallback(result.rows)
            return "Comparison requires two applicants."

        if parsed.intent == IntentType.EXPLAIN_APPLICANT.value:
            if result.success and result.rows:
                prompt = build_explain_prompt(result.rows[0])
                try:
                    response_text = _call_gemini(prompt, active_model)
                    if response_text:
                        return response_text
                except Exception as exc:
                    logger.error("[EXPLAIN] Gemini explain failed: %s", exc)
                return build_explain_fallback(result.rows[0])
            return "No data found for the requested applicant."

        if parsed.intent == IntentType.GENERAL_QUESTION.value:
            return handle_general_question(raw_query, active_model)

        if active_model is None:
            if parsed.intent == IntentType.AGGREGATE_QUERY.value:
                return build_aggregate_fallback(result)
            return format_results_as_text(result)

        prompt = build_response_prompt(parsed, result, raw_query)
        try:
            response_text = _call_gemini(prompt, active_model)
            if response_text:
                return response_text
        except Exception:
            logger.exception("Failed to generate general natural-language response")
            if parsed.intent == IntentType.AGGREGATE_QUERY.value:
                return build_aggregate_fallback(result)
        if parsed.intent == IntentType.AGGREGATE_QUERY.value:
            return build_aggregate_fallback(result)
        return format_results_as_text(result)
    except Exception:
        logger.exception("Failed to generate natural-language response")
        return format_results_as_text(result)


def handle_general_question(query: str, gemini_model: Any) -> str:
    """Answer a general credit-scoring question without DB data."""

    prompt = (
        "You are a credit analyst. Answer this question about credit scoring concisely: "
        f"{query}"
    )

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


if __name__ == "__main__":
    sample_row = {
        "SK_ID_CURR": 123654,
        "prediction": "REJECT",
        "credit_score": 420,
        "risk_band": "HIGH",
        "probability": 0.73,
        "AMT_INCOME_TOTAL_CAPPED": 67500,
        "AMT_CREDIT": 450000,
        "AGE_YEARS": 34,
        "DAYS_EMPLOYED": -1460,
        "top_features": "high debt-to-income ratio, short employment history, previous late payments",
        "explanation_text": "Applicant shows elevated default risk due to income-to-loan ratio exceeding safe thresholds",
    }
    print(build_explain_prompt(sample_row))
