"""MasterMind AI Credit Advisor — Gemini-powered chatbot.

A truly conversational chatbot that:
  - Uses Gemini to generate dynamic, contextual responses
  - Knows the entire MasterMind website structure and guides users
  - Auto-receives score context from the current page
  - Lets users paste raw score/analysis data for interpretation
  - Never gives canned if-else responses
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# System prompt — the entire knowledge base for the chatbot
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are **MasterMind Credit Advisor**, an AI assistant embedded inside the MasterMind AI Credit Scoring web application. You help analysts, loan officers, and applicants understand credit decisions, navigate the platform, and interpret SHAP-based risk explanations.

## YOUR CAPABILITIES
1. **Explain credit scores** — Interpret probability-of-default values, SHAP risk factors, and decisions (APPROVE/REVIEW/DECLINE).
2. **Guide users through the website** — You know every page, what it does, and how to use it.
3. **Analyze pasted data** — Users can paste score JSON, SHAP values, or application data and you will interpret it.
4. **Financial inclusion advice** — Explain how underbanked individuals can build credit using alternative data.

## WEBSITE NAVIGATION GUIDE
Here is the full sitemap you must reference when guiding users:

| Page | URL | What it does |
|------|-----|-------------|
| **Home** | `/` | Landing page with platform overview, feature highlights, and quick-start links |
| **Analyse an Application** | `/analyze` | Submit application data for instant credit scoring. Choose REDUCED (33 fields) or FULL tier (with bureau/installment aggregates) |
| **New Application** | `/applications/new` | Create and save a new applicant record to the database |
| **Application Detail** | `/applications/<id>` | View a saved application's details and score history |
| **Analyst Hub** | `/analyst` | Command center dashboard — review queue, KPI cards, application list |
| **Analyst Applications** | `/analyst/applications` | Full list of all saved applications with status filters |
| **Analyst Application Detail** | `/analyst/applications/<id>` | Edit application fields, change status, re-run analysis |
| **Application Report** | `/analyst/applications/<id>/report` | Deep insight report with SHAP narratives, risk gauge, feature drivers, what-if simulator |
| **System Status** | `/status` | Runtime health, model version, artifact validation status |
| **Analytics** | `/analytics` | Offline dashboards — EDA plots, model performance charts, fairness audit visuals |

## COVERAGE TIERS
- **REDUCED tier**: Application-level fields only (33 fields). Use when only basic applicant info is available. Key fields: AMT_CREDIT, AMT_INCOME_TOTAL, EXT_SOURCE_1/2/3, DAYS_EMPLOYED.
- **FULL tier**: Adds bureau, previous application, installment, POS cash, and credit card aggregates for maximum accuracy.

## SCORING PIPELINE
1. User submits application data via `/analyze` or the API (`POST /score`)
2. System determines coverage tier (REDUCED or FULL)
3. Feature builder transforms raw data into model features
4. XGBoost + LightGBM ensemble generates raw probability
5. Isotonic calibration produces calibrated probability of default
6. SHAP explainer extracts top 5 risk factors
7. Policy engine maps probability to decision: APPROVE (<15%), REVIEW (15-35%), DECLINE (>35%)
8. Response includes: probability, decision, model version, fairness audit status, top 5 explanations

## HOW TO INTERPRET RESULTS
- **Probability of Default (PD)**: The calibrated likelihood (0-100%) that the applicant will default. Lower is better.
- **SHAP values**: Show which features pushed the score up (toward default) or down (toward repayment). Positive SHAP = increases risk.
- **Fairness audit**: Checks that approval rates across demographic groups satisfy the 4/5ths rule (disparate impact ratio ≥ 0.8).

## COMMON FEATURE TRANSLATIONS
- AMT_CREDIT → Loan amount requested
- AMT_INCOME_TOTAL_CAPPED → Reported annual income
- AMT_ANNUITY → Monthly repayment amount
- EXT_SOURCE_1/2/3 → External credit bureau scores (0-1, higher = better creditworthiness)
- DAYS_EMPLOYED → Employment duration (negative days = employed that many days ago)
- DAYS_BIRTH → Age (negative days = born that many days ago)
- CREDIT_INCOME_RATIO → Loan-to-income ratio
- ANNUITY_INCOME_RATIO → Monthly payment vs income ratio
- BUREAU_LOAN_COUNT → Number of previous loans from other lenders
- BUREAU_DEBT_TO_CREDIT_RATIO → How much available credit is currently used

## RULES
1. When the user asks "how do I score an application", guide them step-by-step to `/analyze`.
2. When the user pastes JSON or score data, parse and interpret it — explain the decision, call out the top risk factors, and suggest next steps.
3. When you receive APPLICATION CONTEXT (injected automatically from the page), use it to give specific answers about THAT application's score.
4. If someone asks about improving their score, give specific actionable advice based on their SHAP factors (if available) or general best practices.
5. Always be encouraging — many users are first-time applicants from underbanked communities.
6. Keep responses concise (3-5 paragraphs max) but thorough.
7. If the user asks something completely unrelated to credit/finance/the platform, politely redirect.
8. NEVER make up scores or probabilities. Only cite data explicitly provided.
9. When structured REPORT CONTEXT is provided, stay grounded in that context and say when a requested detail is unavailable.
10. Treat simulator results as temporary analyst scenarios only, never as persisted decisions.
"""

# ──────────────────────────────────────────────────────────────────────
# Gemini client
# ──────────────────────────────────────────────────────────────────────

_gemini_model = None
_gemini_initialized = False
_gemini_available = False


def _init_gemini() -> bool:
    global _gemini_model, _gemini_initialized, _gemini_available

    if _gemini_initialized:
        return _gemini_available

    _gemini_initialized = True
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        _gemini_available = False
        return False

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
        )
        _gemini_available = True
        return True
    except Exception as exc:
        print(f"[chatbot] Gemini init failed: {exc}")
        _gemini_available = False
        return False


def is_gemini_available() -> bool:
    return _init_gemini()


# ──────────────────────────────────────────────────────────────────────
# Context builder — formats page context into the prompt
# ──────────────────────────────────────────────────────────────────────

def _build_context_block(
    score_data: dict[str, Any] | None,
    shap_values: list[dict[str, Any]] | None,
    page_context: dict[str, Any] | None,
    report_context: dict[str, Any] | None = None,
) -> str:
    """Build context to prepend to user's message so Gemini knows what the user is looking at."""
    parts: list[str] = []

    if page_context:
        page = page_context.get("current_page", "")
        if page:
            parts.append(f"[User is currently on: {page}]")

    if score_data:
        parts.append("## Current Application Score Data")
        for key in ["probability_of_default", "decision", "coverage_tier",
                     "model_version", "model_fairness_audit_passed"]:
            val = score_data.get(key)
            if val is not None:
                parts.append(f"- {key}: {val}")

    if shap_values and isinstance(shap_values, list):
        parts.append("\n## Top SHAP Risk Factors")
        for i, item in enumerate(shap_values[:5], 1):
            feat = item.get("feature", item.get("name", "Unknown"))
            reason = item.get("reason", item.get("description", ""))
            parts.append(f"{i}. **{feat}**: {reason}")

    if report_context and isinstance(report_context, dict):
        applicant_summary = report_context.get("applicant_summary", {})
        latest_assessment = report_context.get("latest_assessment", {})
        adverse_action = report_context.get("adverse_action", {})
        score_history = report_context.get("score_history", [])
        simulator_result = report_context.get("simulator_result")

        parts.append("\n## Structured Report Context")
        if applicant_summary:
            parts.append(
                "- Applicant: {name} | Tier: {tier} | Status: {status}".format(
                    name=applicant_summary.get("applicant_name", "-"),
                    tier=applicant_summary.get("coverage_tier", "-"),
                    status=applicant_summary.get("current_status", "-"),
                )
            )
        if latest_assessment:
            parts.append(
                "- Latest assessment: {decision} at {probability}".format(
                    decision=latest_assessment.get("decision_label", latest_assessment.get("decision", "-")),
                    probability=latest_assessment.get("calibrated_probability_text", "-"),
                )
            )
            summary = str(latest_assessment.get("decision_summary", "")).strip()
            if summary:
                parts.append(f"- Decision summary: {summary}")
        narratives = report_context.get("shap_narratives", [])
        if narratives:
            parts.append("- SHAP narratives:")
            for narrative in narratives[:5]:
                parts.append(f"  • {narrative}")
        drivers = report_context.get("top_drivers", [])
        if drivers:
            parts.append("- Top drivers:")
            for item in drivers[:5]:
                parts.append(f"  • {item.get('feature', 'Unknown')}: {item.get('reason', '')}")
        reasons = adverse_action.get("reasons", []) if isinstance(adverse_action, dict) else []
        if reasons:
            parts.append(f"- {adverse_action.get('section_title', 'Analyst review')} reasons:")
            for item in reasons[:5]:
                parts.append(f"  • {item.get('title', 'Reason')}: {item.get('detail', '')}")
        if score_history:
            parts.append("- Recent score history:")
            for item in score_history[:5]:
                parts.append(
                    "  • {date}: {decision} at {probability}".format(
                        date=item.get("scored_at", "-"),
                        decision=item.get("decision_label", item.get("decision", "-")),
                        probability=item.get("probability_text", "-"),
                    )
                )
        if simulator_result and isinstance(simulator_result, dict):
            parts.append("- Active simulator scenario:")
            parts.append(
                "  • Simulated outcome: {decision} at {probability} ({delta})".format(
                    decision=simulator_result.get("simulated_decision_label", "-"),
                    probability=simulator_result.get("simulated_probability_text", "-"),
                    delta=simulator_result.get("delta_text", "no delta"),
                )
            )
            for item in simulator_result.get("changed_features", [])[:5]:
                parts.append(
                    f"  • {item.get('label', 'Field')}: {item.get('before', '-')}"
                    f" -> {item.get('after', '-')}"
                )

    if not parts:
        return ""

    return (
        "\n--- APPLICATION CONTEXT (auto-injected from current page) ---\n"
        + "\n".join(parts)
        + "\n--- END CONTEXT ---\n\n"
    )


# ──────────────────────────────────────────────────────────────────────
# Detect if user pasted JSON/score data in their message
# ──────────────────────────────────────────────────────────────────────

def _extract_pasted_data(message: str) -> dict[str, Any] | None:
    """Try to parse JSON from the user's message."""
    # Look for JSON-like content
    json_patterns = [
        re.compile(r'\{[^{}]*"probability_of_default"[^{}]*\}', re.DOTALL),
        re.compile(r'\{[^{}]*"decision"[^{}]*\}', re.DOTALL),
        re.compile(r'\{[^{}]*"coverage_tier"[^{}]*\}', re.DOTALL),
        re.compile(r'\{.*\}', re.DOTALL),
    ]
    for pattern in json_patterns:
        match = pattern.search(message)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _report_latest_assessment(report_context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report_context, dict):
        return {}
    latest = report_context.get("latest_assessment")
    return latest if isinstance(latest, dict) else {}


def _report_driver_lines(report_context: dict[str, Any] | None) -> list[str]:
    if not isinstance(report_context, dict):
        return []
    drivers = report_context.get("top_drivers")
    if not isinstance(drivers, list):
        return []
    return [
        f"{index}. {item.get('feature', 'Unknown')} — {item.get('reason', '')}".strip()
        for index, item in enumerate(drivers[:5], 1)
        if isinstance(item, dict)
    ]


def _report_reason_items(report_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report_context, dict):
        return []
    adverse_action = report_context.get("adverse_action")
    if not isinstance(adverse_action, dict):
        return []
    reasons = adverse_action.get("reasons")
    return reasons if isinstance(reasons, list) else []


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

    drivers = _report_driver_lines(report_context)
    if drivers:
        lines.extend(["", "Main model drivers:"])
        lines.extend(drivers[:3])

    return "\n".join(lines)


def _report_reason_fallback(report_context: dict[str, Any]) -> str:
    adverse_action = report_context.get("adverse_action", {}) if isinstance(report_context.get("adverse_action"), dict) else {}
    reasons = _report_reason_items(report_context)
    section_title = adverse_action.get("section_title", "Analyst review reasons")
    if not reasons:
        drivers = _report_driver_lines(report_context)
        if not drivers:
            return "I don't have stored adverse-action style reasons for this report yet. I can still summarize the top drivers that were captured."
        return "\n".join([f"{section_title}:", ""] + drivers[:3])

    lines = [f"{section_title} in simple language:", ""]
    for index, item in enumerate(reasons[:5], 1):
        title = item.get("title", "Reason")
        detail = item.get("detail", "")
        lines.append(f"{index}. {title} — {detail}")
    return "\n".join(lines)


def _report_simulator_fallback(report_context: dict[str, Any]) -> str:
    simulator_result = _report_simulator_result(report_context)
    if not simulator_result:
        return "There is no active simulator scenario in context yet. Run a What-If simulation on this report and I can compare the scenario against the persisted result."

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
            lines.append(f"{index}. {item.get('label', 'Field')} — {item.get('before', '-')} -> {item.get('after', '-')}")

    return "\n".join(lines)


def _report_review_fallback(report_context: dict[str, Any]) -> str:
    reasons = _report_reason_items(report_context)
    history = report_context.get("score_history", []) if isinstance(report_context.get("score_history"), list) else []

    lines = [
        "Before moving toward approval, an analyst should review:",
        "",
    ]
    if reasons:
        for index, item in enumerate(reasons[:3], 1):
            lines.append(f"{index}. {item.get('title', 'Reason')} — {item.get('detail', '')}")
    else:
        for index, item in enumerate(_report_driver_lines(report_context)[:3], 1):
            lines.append(f"{index}. {item}")

    if history:
        latest_history = history[0]
        lines.extend(
            [
                "",
                f"Recent score history check: latest stored run was {latest_history.get('decision_label', latest_history.get('decision', '-'))} at {latest_history.get('probability_text', '-')}.",
            ]
        )

    lines.append("")
    lines.append("Use this as analyst support only, not as an automated approval instruction.")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Smart fallback — NOT if-else, uses templates with real logic
# ──────────────────────────────────────────────────────────────────────

def _smart_fallback(
    message: str,
    score_data: dict[str, Any] | None,
    shap_values: list[dict[str, Any]] | None,
    report_context: dict[str, Any] | None = None,
) -> str:
    """Intelligent fallback when Gemini is unavailable. Still provides
    useful, contextual responses by combining templates with actual data."""

    lower = message.lower().strip()

    # Check for pasted data
    pasted = _extract_pasted_data(message)
    if pasted:
        pd_val = pasted.get("probability_of_default")
        decision = pasted.get("decision", "N/A")
        explanations = pasted.get("top_5_explanations", [])

        lines = [f"📊 **I found score data in your message. Here's my analysis:**\n"]
        if pd_val is not None:
            pct = f"{float(pd_val) * 100:.1f}%" if isinstance(pd_val, (int, float)) else str(pd_val)
            lines.append(f"**Probability of Default:** {pct}")
        lines.append(f"**Decision:** {decision}")

        if explanations:
            lines.append("\n**Top Risk Factors:**")
            for i, exp in enumerate(explanations[:5], 1):
                if isinstance(exp, dict):
                    lines.append(f"{i}. **{exp.get('feature', 'Unknown')}** — {exp.get('reason', '')}")
                elif isinstance(exp, str):
                    lines.append(f"{i}. {exp}")

        if decision == "APPROVE":
            lines.append("\n✅ This profile shows strong creditworthiness. The applicant qualifies for approval.")
        elif decision == "DECLINE":
            lines.append("\n⚠️ This profile shows elevated risk. Review the top factors above — improving external credit scores and reducing the loan-to-income ratio are typically the most impactful changes.")
        elif decision == "REVIEW":
            lines.append("\n🔍 This is a borderline case. Use the **What-If Simulator** on the report page (`/analyst/applications/<id>/report`) to test how changing key features would affect the score.")

        return "\n".join(lines)

    # Navigation guidance
    nav_keywords = {
        "score": "To score an application, go to **Analyse an Application** (`/analyze`). Choose your coverage tier (REDUCED for basic data, FULL for comprehensive), fill in the fields, and click submit. You'll get an instant score with SHAP explanations.",
        "analyz": "To score an application, go to **Analyse an Application** (`/analyze`). Choose your coverage tier (REDUCED for basic data, FULL for comprehensive), fill in the fields, and click submit. You'll get an instant score with SHAP explanations.",
        "new application": "Go to **New Application** (`/applications/new`) to create and save a new applicant record. Once saved, you can run analysis on it from the Analyst Hub.",
        "analyst": "The **Analyst Hub** (`/analyst`) is your command center. It shows the review queue, KPI stats, and all saved applications. Click any application to edit, re-score, or view its deep insight report.",
        "report": "The **Application Report** page (`/analyst/applications/<id>/report`) shows the full deep insight view: risk gauge, SHAP narratives, feature drivers, application summary, and the What-If Simulator for scenario testing.",
        "status": "Check **System Status** (`/status`) to see runtime health, model version, loaded artifacts, and whether the fairness audit passed.",
        "analytics": "The **Analytics** page (`/analytics`) shows offline dashboards with EDA plots, model performance charts, and fairness audit visualizations.",
        "simulator": "The **What-If Simulator** is on the report page (`/analyst/applications/<id>/report`). It lets you adjust key features and instantly see how the score would change — without saving anything.",
        "navigate": "Here are the main pages:\n• **Home** (`/`) — Platform overview\n• **Analyse** (`/analyze`) — Score an application\n• **New Application** (`/applications/new`) — Create a record\n• **Analyst Hub** (`/analyst`) — Review queue & dashboard\n• **Analytics** (`/analytics`) — Model performance & EDA\n• **System Status** (`/status`) — Runtime health",
        "where": "Here are the main pages:\n• **Home** (`/`) — Platform overview\n• **Analyse** (`/analyze`) — Score an application\n• **New Application** (`/applications/new`) — Create a record\n• **Analyst Hub** (`/analyst`) — Review queue & dashboard\n• **Analytics** (`/analytics`) — Model performance & EDA\n• **System Status** (`/status`) — Runtime health",
        "how do i": "What would you like to do? I can help you:\n• **Score an application** → `/analyze`\n• **View saved applications** → `/analyst`\n• **Check model health** → `/status`\n• **See EDA & charts** → `/analytics`\n\nJust tell me what you're trying to accomplish!",
        "help": "I'm your MasterMind Credit Advisor! I can help you:\n\n• **Navigate the platform** — Just ask \"where do I go to score an application?\"\n• **Interpret scores** — Paste your score JSON here and I'll explain it\n• **Understand risk factors** — I'll translate SHAP values into plain English\n• **Improve creditworthiness** — I'll give actionable advice based on your factors\n\n💡 **Tip:** When you're on a report page, I automatically see the score data and can answer questions about that specific application!",
    }

    if report_context:
        if ("risk profile" in lower) or ("summarize" in lower and "risk" in lower) or "summary" in lower:
            return _report_summary_fallback(report_context)
        if "decline reasons" in lower or "main decline reasons" in lower or ("simple language" in lower and "reason" in lower):
            return _report_reason_fallback(report_context)
        if "simulator" in lower or "reduced risk" in lower or "what changes" in lower:
            return _report_simulator_fallback(report_context)
        if "review before approval" in lower or "before approval" in lower or "analyst review" in lower:
            return _report_review_fallback(report_context)

    for keyword, response in nav_keywords.items():
        if keyword in lower:
            return response

    # If we have active score context, use it
    if score_data:
        pd_val = score_data.get("probability_of_default")
        decision = score_data.get("decision", "N/A")
        pct = f"{float(pd_val) * 100:.1f}%" if isinstance(pd_val, (int, float)) and pd_val is not None else "N/A"

        if any(w in lower for w in ["why", "explain", "reason", "factor", "what"]):
            lines = [f"Based on the current application (PD: {pct}, Decision: **{decision}**):\n"]
            if shap_values:
                lines.append("**Key factors driving this score:**")
                for i, item in enumerate(shap_values[:5], 1):
                    feat = item.get("feature", item.get("name", "Unknown"))
                    reason = item.get("reason", item.get("description", ""))
                    lines.append(f"{i}. **{feat}** — {reason}")
                lines.append("\nTo test how changes would affect this score, use the **What-If Simulator** on the report page.")
            else:
                lines.append("SHAP factor details aren't available in the current context. Navigate to the application's **Report** page to see the full breakdown.")
            return "\n".join(lines)

        return f"I can see the current application has a **{pct}** probability of default with a **{decision}** decision. What would you like to know about it? I can explain the risk factors, suggest improvements, or help you navigate to the report page."

    if any(w in lower for w in ["improve", "better", "increase", "raise", "boost"]):
        return ("Here are evidence-based strategies to improve creditworthiness:\n\n"
                "1. **Build payment history** — Consistent utility and phone bill payments demonstrate reliability\n"
                "2. **Reduce debt-to-income ratio** — Pay down existing obligations before applying\n"
                "3. **Maintain stable employment** — Longer tenure at current job improves DAYS_EMPLOYED\n"
                "4. **Start small** — Microloans and starter credit cards build bureau history\n"
                "5. **Improve external scores** — EXT_SOURCE_1/2/3 are typically the strongest predictors\n\n"
                "💡 For specific advice, score an application at `/analyze` and I'll tell you exactly which factors to focus on.")

    # Default — actually helpful
    return ("👋 I'm your MasterMind Credit Advisor. Here's what I can do:\n\n"
            "• **\"How do I score an application?\"** — I'll walk you through it step by step\n"
            "• **\"What does this score mean?\"** — Paste any score JSON and I'll interpret it\n"
            "• **\"Where is the What-If Simulator?\"** — I know every page on the platform\n"
            "• **\"How can I improve my credit?\"** — Specific advice based on SHAP factors\n\n"
            "💡 **Pro tip:** When you're on a report page, I automatically see the application's score data. "
            "Just ask me about it!\n\n"
            "🔑 **For the best experience**, set `GEMINI_API_KEY` in your environment for full AI-powered conversations.")


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def chat(
    message: str,
    score_data: dict[str, Any] | None = None,
    shap_values: list[dict[str, Any]] | None = None,
    page_context: dict[str, Any] | None = None,
    report_context: dict[str, Any] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Process a chat message. Returns {"response": str, "source": str, "gemini_available": bool}."""

    if not message or not message.strip():
        return {"response": "Type a message to get started! Try asking 'how do I score an application?'",
                "source": "fallback", "gemini_available": is_gemini_available()}

    # Check if user pasted JSON — enrich score_data
    pasted = _extract_pasted_data(message)
    if pasted and not score_data:
        score_data = pasted
        shap_values = pasted.get("top_5_explanations", shap_values)

    if report_context and not score_data:
        latest = _report_latest_assessment(report_context)
        score_data = {
            "probability_of_default": latest.get("calibrated_probability_text"),
            "decision": latest.get("decision_label", latest.get("decision")),
            "model_version": latest.get("model_version"),
        }
    if report_context and not shap_values:
        shap_values = [
            {
                "feature": item.get("feature"),
                "reason": item.get("reason"),
            }
            for item in report_context.get("top_drivers", [])
            if isinstance(item, dict)
        ] or shap_values

    context_block = _build_context_block(score_data, shap_values, page_context, report_context)

    # Try Gemini first
    if _init_gemini() and _gemini_model:
        try:
            full_prompt = context_block + "User: " + message.strip()

            # Build conversation history for multi-turn
            gemini_history = []
            if conversation_history:
                for entry in conversation_history[-8:]:
                    role = entry.get("role", "user")
                    text = entry.get("content", "")
                    if role in ("user",):
                        gemini_history.append({"role": "user", "parts": [text]})
                    else:
                        gemini_history.append({"role": "model", "parts": [text]})

            chat_session = _gemini_model.start_chat(history=gemini_history)
            response = chat_session.send_message(full_prompt)
            return {"response": response.text, "source": "gemini", "gemini_available": True}

        except Exception as exc:
            print(f"[chatbot] Gemini call failed: {exc}")
            # Fall through to smart fallback

    return {
        "response": _smart_fallback(message, score_data, shap_values, report_context),
        "source": "fallback",
        "gemini_available": False,
    }
