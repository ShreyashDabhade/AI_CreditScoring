"""Chatbot response layer converting structured results to human text via Gemini.

Provides:
- generate_chatbot_response(user_query: str, structured_data: dict) -> str

This module prefers using the Google Generative API (Gemini) when available
and falls back to a deterministic, non-hallucinating renderer when not.
"""
from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict

from gemini_client import get_model

logger = logging.getLogger(__name__)

# Lightweight in-memory chat history (optional, non-persistent)
chat_history: list[dict] = []


def _safe_to_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return ""


def _deterministic_render(user_query: str, structured: Dict[str, Any]) -> str:
    """Produce a conservative human-readable summary from structured data.

    This renderer never fabricates facts and uses only the provided data.
    """
    if not structured:
        return "No matching applicants found."

    # If the structured payload explicitly contains an error or empty results
    status = structured.get("status")
    results = structured.get("results") or structured.get("applicants") or []
    explanation = structured.get("explanation")

    if status == "error":
        return "I cannot answer that query."

    if not results:
        return "No matching applicants found."

    # Single applicant summary
    if isinstance(results, list) and len(results) == 1:
        r = results[0]
        aid = r.get("applicant_id") or r.get("id") or "<unknown>"
        prob = r.get("probability")
        prob_text = f"their probability of default is {round(float(prob)*100,1)}%" if prob not in (None, "") else "their probability of default is unknown"
        reasons = None
        top = r.get("top_features") or []
        if isinstance(top, str):
            try:
                top = json.loads(top)
            except Exception:
                top = []
        if top:
            reasons = ", ".join([str(t.get("feature")) for t in top[:3] if isinstance(t, dict) and t.get("feature")])

        parts = [f"Applicant {aid} was found.", prob_text + "."]
        if reasons:
            parts.append(f"Top drivers include: {reasons}.")
        return " ".join(parts)

    # Comparison or multiple results
    if isinstance(results, list) and len(results) > 1:
        # If comparison info available, use it; otherwise give counts
        comp = structured.get("comparison")
        if comp and isinstance(comp, dict):
            # Summarize keys in comparison dict conservatively
            diffs = []
            for k, v in comp.items():
                diffs.append(f"{k}: {v}")
            return "Comparison summary: " + "; ".join(diffs)
        return f"Found {len(results)} applicants matching your query."

    # Fallback
    return "No matching applicants found."


def generate_chatbot_response(user_query: str, structured_data: Dict[str, Any]) -> str:
    """Generate a human-friendly response from structured data.

    Attempts to call Gemini (via `google.generativeai`) if `GEMINI_API_KEY`
    is present. Otherwise uses a deterministic local renderer.
    """
    # Append to minimal chat history (non-persistent)
    try:
        chat_history.append({"user": user_query, "data_snapshot": structured_data})
    except Exception:
        pass

    system_prompt = (
        "You are a financial risk assistant for a bank manager.\n"
        "Explain the provided data concisely in business language."
    )

    prompt = (
        "User query:\n"
        f"{user_query}\n\n"
        "Data:\n"
        f"{_safe_to_json(structured_data)}\n\n"
        "Instructions:\n"
        "- Explain results in simple business language\n"
        "- Be concise and clear\n"
        "- Do NOT hallucinate\n"
        "- Only use given data\n"
        "- If comparing, highlight key differences\n"
    )

    try:
        model = get_model()
        logger.info("[CHATBOT] Using shared Gemini model for response generation")
        try:
            resp = model.generate_content(system_prompt + "\n\n" + prompt)
        except TypeError:
            resp = model.generate_content(prompt=system_prompt + "\n\n" + prompt)

        text = None
        if resp is not None:
            text = getattr(resp, "text", None) or getattr(resp, "output", None) or getattr(resp, "content", None)
            if not text and hasattr(resp, "candidates"):
                try:
                    text = "\n".join([getattr(c, "output", str(c)) for c in resp.candidates])
                except Exception:
                    text = str(resp)
        if text:
            chat_history.append({"assistant": text})
            return text.strip()
    except Exception:
        logger.exception("Unexpected error while calling Gemini from chatbot_engine")

    # deterministic fallback
    try:
        text = _deterministic_render(user_query, structured_data)
        try:
            chat_history.append({"assistant": text})
        except Exception:
            pass
        return text
    except Exception:
        return "No matching applicants found."
