from __future__ import annotations

import src.chatbot as chatbot


def _disable_gemini(monkeypatch):
    monkeypatch.setattr(chatbot, "_gemini_initialized", True)
    monkeypatch.setattr(chatbot, "_gemini_available", False)
    monkeypatch.setattr(chatbot, "_gemini_model", None)


def _report_context() -> dict:
    return {
        "scope": "analyst_application_report",
        "application_id": 7,
        "applicant_summary": {
            "applicant_name": "Grounded Analyst Case",
            "coverage_tier": "REDUCED",
            "current_status": "REVIEW",
        },
        "latest_assessment": {
            "decision": "REVIEW",
            "decision_label": "Review",
            "calibrated_probability": 0.184,
            "calibrated_probability_text": "18.4%",
            "decision_summary": "The calibrated risk signal requires analyst review before final action.",
            "model_version": "reduced_v2.1.0",
        },
        "top_drivers": [
            {
                "feature": "External Source 2",
                "reason": "External reference signals were weaker than preferred.",
                "rank": 1,
            },
            {
                "feature": "Credit Amount",
                "reason": "Requested credit remains elevated relative to current income.",
                "rank": 2,
            },
        ],
        "shap_narratives": [
            "External Source 2 is the leading risk driver in this assessment because external reference signals were weaker than preferred."
        ],
        "score_history": [
            {
                "scored_at": "29 Mar 2026, 10:45 UTC",
                "decision": "REVIEW",
                "decision_label": "Review",
                "probability_text": "18.4%",
                "model_version": "reduced_v2.1.0",
                "is_latest": True,
            }
        ],
        "adverse_action": {
            "section_title": "Adverse Action Summary",
            "summary": "External reference quality and affordability signals kept this case in review.",
            "reasons": [
                {
                    "title": "External credit reference strength",
                    "detail": "External credit reference signals were weaker than preferred for this policy.",
                },
                {
                    "title": "Loan affordability pressure",
                    "detail": "Requested credit appears high relative to current reported income.",
                },
            ],
        },
        "simulator_result": {
            "original_probability_text": "18.4%",
            "original_decision_label": "Review",
            "simulated_probability_text": "14.9%",
            "simulated_decision_label": "Approve",
            "delta_text": "-3.5 pts",
            "risk_movement_label": "Risk decreased",
            "changed_features": [
                {
                    "label": "Credit Amount",
                    "before": "300,000",
                    "after": "250,000",
                }
            ],
        },
    }


def test_chat_uses_grounded_report_context_for_risk_summary(monkeypatch):
    _disable_gemini(monkeypatch)

    result = chatbot.chat("Summarize the risk profile", report_context=_report_context())

    assert result["source"] == "fallback"
    assert "18.4%" in result["response"]
    assert "External Source 2" in result["response"]
    assert "Grounded Analyst Case" in result["response"]


def test_chat_uses_grounded_report_context_for_simulator_questions(monkeypatch):
    _disable_gemini(monkeypatch)

    result = chatbot.chat("What changes reduced risk in the simulator?", report_context=_report_context())

    assert result["source"] == "fallback"
    assert "14.9%" in result["response"]
    assert "Risk decreased" in result["response"]
    assert "Credit Amount" in result["response"]
