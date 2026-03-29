"""Compatibility adapter for the active chatbot backend.

The frontend contract in this branch stays the same:
`POST /api/chat` with `message`, `context`, and `history`.

Behind that stable surface, chat requests now flow into the newer
DataPipeline-style parser -> SQL -> validate -> execute -> respond pipeline.
"""
from __future__ import annotations

import json
import re
from typing import Any

from gemini_client import is_available as _gemini_is_available

from src.api import agent_routes


def _extract_pasted_data(message: str) -> dict[str, Any] | None:
    text = str(message or "")
    json_patterns = [
        re.compile(r'\{[^{}]*"probability_of_default"[^{}]*\}', re.DOTALL),
        re.compile(r'\{[^{}]*"decision"[^{}]*\}', re.DOTALL),
        re.compile(r'\{.*\}', re.DOTALL),
    ]
    for pattern in json_patterns:
        match = pattern.search(text)
        if not match:
            continue
        try:
            parsed = json.loads(match.group())
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _report_latest_assessment(report_context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report_context, dict):
        return {}
    latest = report_context.get("latest_assessment")
    return latest if isinstance(latest, dict) else {}


def _report_driver_items(report_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report_context, dict):
        return []
    drivers = report_context.get("top_drivers")
    return [item for item in drivers if isinstance(item, dict)] if isinstance(drivers, list) else []


def _report_reason_items(report_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report_context, dict):
        return []
    adverse_action = report_context.get("adverse_action")
    if not isinstance(adverse_action, dict):
        return []
    reasons = adverse_action.get("reasons")
    return [item for item in reasons if isinstance(item, dict)] if isinstance(reasons, list) else []


def _report_simulator_result(report_context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report_context, dict):
        return {}
    simulator_result = report_context.get("simulator_result")
    return simulator_result if isinstance(simulator_result, dict) else {}


def _report_summary_fallback(report_context: dict[str, Any]) -> str:
    applicant = report_context.get("applicant_summary", {}) if isinstance(report_context.get("applicant_summary"), dict) else {}
    latest = _report_latest_assessment(report_context)
    lines = [
        "Here is the grounded risk profile for this application:",
        "",
        f"- Applicant: {applicant.get('applicant_name', '-')}",
        f"- Current workflow status: {applicant.get('current_status', '-')}",
        f"- Latest decision: {latest.get('decision_label', latest.get('decision', '-'))}",
        f"- Calibrated probability of default: {latest.get('calibrated_probability_text', '-')}",
    ]
    summary = str(latest.get("decision_summary", "")).strip()
    if summary:
        lines.append(f"- Decision framing: {summary}")

    drivers = _report_driver_items(report_context)
    if drivers:
        lines.extend(["", "Main model drivers:"])
        for index, item in enumerate(drivers[:3], 1):
            lines.append(f"{index}. {item.get('feature', 'Unknown')} - {item.get('reason', '')}".strip())

    return "\n".join(lines)


def _report_reason_fallback(report_context: dict[str, Any]) -> str:
    adverse_action = report_context.get("adverse_action", {}) if isinstance(report_context.get("adverse_action"), dict) else {}
    reasons = _report_reason_items(report_context)
    section_title = adverse_action.get("section_title", "Analyst review reasons")
    if not reasons:
        drivers = _report_driver_items(report_context)
        if not drivers:
            return "I do not have stored analyst reasons for this report yet."
        lines = [f"{section_title}:", ""]
        for index, item in enumerate(drivers[:3], 1):
            lines.append(f"{index}. {item.get('feature', 'Unknown')} - {item.get('reason', '')}".strip())
        return "\n".join(lines)

    lines = [f"{section_title} in simple language:", ""]
    for index, item in enumerate(reasons[:5], 1):
        lines.append(f"{index}. {item.get('title', 'Reason')} - {item.get('detail', '')}".strip())
    return "\n".join(lines)


def _report_simulator_fallback(report_context: dict[str, Any]) -> str:
    simulator_result = _report_simulator_result(report_context)
    if not simulator_result:
        return "There is no active simulator scenario in context yet."

    lines = [
        "Here is the current simulator readout:",
        "",
        f"- Original outcome: {simulator_result.get('original_decision_label', '-')} at {simulator_result.get('original_probability_text', '-')}",
        f"- Simulated outcome: {simulator_result.get('simulated_decision_label', '-')} at {simulator_result.get('simulated_probability_text', '-')}",
        f"- Probability delta: {simulator_result.get('delta_text', '-')}",
        f"- Risk movement: {simulator_result.get('risk_movement_label', '-')}",
    ]
    changes = simulator_result.get("changed_features")
    if isinstance(changes, list) and changes:
        lines.extend(["", "Fields changed in the scenario:"])
        for index, item in enumerate(changes[:5], 1):
            if not isinstance(item, dict):
                continue
            lines.append(f"{index}. {item.get('label', 'Field')} - {item.get('before', '-')} -> {item.get('after', '-')}")
    return "\n".join(lines)


def _report_review_fallback(report_context: dict[str, Any]) -> str:
    reasons = _report_reason_items(report_context)
    history = report_context.get("score_history", []) if isinstance(report_context.get("score_history"), list) else []

    lines = ["Before moving toward approval, an analyst should review:", ""]
    if reasons:
        for index, item in enumerate(reasons[:3], 1):
            lines.append(f"{index}. {item.get('title', 'Reason')} - {item.get('detail', '')}".strip())
    else:
        for index, item in enumerate(_report_driver_items(report_context)[:3], 1):
            lines.append(f"{index}. {item.get('feature', 'Unknown')} - {item.get('reason', '')}".strip())

    if history:
        latest_history = history[0] if isinstance(history[0], dict) else {}
        lines.extend(
            [
                "",
                "Recent score history check: latest stored run was "
                f"{latest_history.get('decision_label', latest_history.get('decision', '-'))} "
                f"at {latest_history.get('probability_text', '-')}.",
            ]
        )

    lines.extend(["", "Use this as analyst support only, not as an automated approval instruction."])
    return "\n".join(lines)


def _score_context_reply(
    message: str,
    score_data: dict[str, Any] | None,
    shap_values: list[dict[str, Any]] | None,
) -> str | None:
    if not isinstance(score_data, dict):
        return None

    lower = str(message or "").lower()
    probability = score_data.get("probability_of_default")
    decision = score_data.get("decision", "N/A")

    if isinstance(probability, (int, float)):
        pct = f"{float(probability) * 100:.1f}%"
    else:
        pct = str(probability or "N/A")

    if any(token in lower for token in ("why", "explain", "reason", "factor", "score", "risk", "this application")):
        lines = [f"The current application is at {pct} probability of default with a {decision} decision."]
        if isinstance(shap_values, list) and shap_values:
            lines.append("")
            lines.append("Main score drivers:")
            for index, item in enumerate(shap_values[:5], 1):
                if not isinstance(item, dict):
                    continue
                lines.append(f"{index}. {item.get('feature', 'Unknown')} - {item.get('reason', item.get('description', ''))}".strip())
        return "\n".join(lines)

    return None


def _contextual_reply(
    message: str,
    score_data: dict[str, Any] | None,
    shap_values: list[dict[str, Any]] | None,
    report_context: dict[str, Any] | None,
) -> str | None:
    lower = str(message or "").lower().strip()

    if isinstance(report_context, dict):
        if ("risk profile" in lower) or ("summarize" in lower and "risk" in lower) or "summary" in lower:
            return _report_summary_fallback(report_context)
        if "decline reasons" in lower or "main decline reasons" in lower or ("simple language" in lower and "reason" in lower):
            return _report_reason_fallback(report_context)
        if "simulator" in lower or "reduced risk" in lower or "what changes" in lower:
            return _report_simulator_fallback(report_context)
        if "review before approval" in lower or "before approval" in lower or "analyst review" in lower:
            return _report_review_fallback(report_context)

    return _score_context_reply(message, score_data, shap_values)


def is_gemini_available() -> bool:
    return _gemini_is_available()


def _normalize_pipeline_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if not str(normalized.get("response", "")).strip():
        normalized["response"] = (
            str(normalized.get("message") or "").strip()
            or str(normalized.get("error") or "").strip()
            or "Sorry, I could not process that."
        )
    normalized.setdefault("source", "fallback")
    normalized.pop("status_code", None)
    return normalized


def chat(
    message: str,
    score_data: dict[str, Any] | None = None,
    shap_values: list[dict[str, Any]] | None = None,
    page_context: dict[str, Any] | None = None,
    report_context: dict[str, Any] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if not str(message or "").strip():
        return {
            "response": "Type a message to get started.",
            "source": "fallback",
            "gemini_available": is_gemini_available(),
        }

    pasted = _extract_pasted_data(message)
    if pasted and not score_data:
        score_data = pasted
        pasted_explanations = pasted.get("top_5_explanations")
        if isinstance(pasted_explanations, list) and not shap_values:
            shap_values = pasted_explanations

    grounded_reply = _contextual_reply(message, score_data, shap_values, report_context)
    if grounded_reply:
        return {
            "response": grounded_reply,
            "source": "fallback",
            "gemini_available": is_gemini_available(),
        }

    pipeline_payload = agent_routes.run_pipeline(
        message,
        score_data=score_data,
        shap_values=shap_values,
        page_context=page_context,
        report_context=report_context,
        conversation_history=conversation_history,
    )
    pipeline_payload = _normalize_pipeline_payload(pipeline_payload)
    pipeline_payload["gemini_available"] = is_gemini_available()
    return pipeline_payload
