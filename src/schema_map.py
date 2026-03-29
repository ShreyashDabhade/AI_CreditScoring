"""Schema map for the active chatbot applicants projection."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

COLUMN_ALIASES: Dict[str, str] = {
    "application id": "application_id",
    "application": "application_id",
    "applicant": "SK_ID_CURR",
    "applicant id": "SK_ID_CURR",
    "customer id": "SK_ID_CURR",
    "sk_id": "SK_ID_CURR",
    "sk id": "SK_ID_CURR",
    "sk_id_curr": "SK_ID_CURR",
    "income": "AMT_INCOME_TOTAL_CAPPED",
    "salary": "AMT_INCOME_TOTAL_CAPPED",
    "annual income": "AMT_INCOME_TOTAL_CAPPED",
    "loan amount": "AMT_CREDIT",
    "credit amount": "AMT_CREDIT",
    "annuity": "AMT_ANNUITY",
    "age": "AGE_YEARS",
    "gender": "CODE_GENDER",
    "education": "NAME_EDUCATION_TYPE",
    "family members": "CNT_FAM_MEMBERS",
    "employment": "DAYS_EMPLOYED",
    "score": "credit_score",
    "credit score": "credit_score",
    "risk": "risk_band",
    "risk band": "risk_band",
    "decision": "prediction",
    "status": "prediction",
    "result": "prediction",
    "probability": "probability",
    "default probability": "probability",
    "default risk": "probability",
    "top features": "top_features",
    "explanation": "explanation_text",
    "income ratio": "CREDIT_INCOME_RATIO",
    "debt to income": "CREDIT_INCOME_RATIO",
    "external score": "EXT_SOURCE_MEAN",
    "late payments": "INST_LATE_COUNT",
    "coverage tier": "coverage_tier",
    "model version": "model_version",
    "applicant name": "applicant_name",
}

SCHEMA_COLUMNS: List[Dict[str, str]] = [
    {"column": "application_id", "type": "numeric", "description": "Internal application record id.", "example": 7},
    {"column": "SK_ID_CURR", "type": "numeric", "description": "Applicant identifier when available.", "example": 100038},
    {"column": "applicant_id", "type": "numeric", "description": "Compatibility alias for applicant id.", "example": 100038},
    {"column": "applicant_name", "type": "text", "description": "Applicant display name.", "example": "Jordan Review"},
    {"column": "AMT_INCOME_TOTAL_CAPPED", "type": "numeric", "description": "Annual income.", "example": 90000.0},
    {"column": "AMT_CREDIT", "type": "numeric", "description": "Requested credit amount.", "example": 250000.0},
    {"column": "AMT_ANNUITY", "type": "numeric", "description": "Annuity or repayment amount.", "example": 25000.0},
    {"column": "AGE_YEARS", "type": "numeric", "description": "Applicant age in years.", "example": 34},
    {"column": "CODE_GENDER", "type": "text", "description": "Gender code.", "example": "M"},
    {"column": "NAME_EDUCATION_TYPE", "type": "text", "description": "Education level.", "example": "Higher education"},
    {"column": "CNT_FAM_MEMBERS", "type": "numeric", "description": "Family member count.", "example": 3},
    {"column": "DAYS_EMPLOYED", "type": "numeric", "description": "Employment duration in days.", "example": -1460},
    {"column": "prediction", "type": "text", "description": "Decision output such as APPROVE, REVIEW, or DECLINE.", "example": "REVIEW"},
    {"column": "credit_score", "type": "numeric", "description": "Derived score-style number for conversational summaries.", "example": 640},
    {"column": "risk_band", "type": "text", "description": "Risk label such as LOW, MEDIUM, or HIGH.", "example": "MEDIUM"},
    {"column": "probability", "type": "numeric", "description": "Probability of default between 0 and 1.", "example": 0.184},
    {"column": "top_features", "type": "text", "description": "Serialized top model drivers.", "example": "[{\"feature\": \"Credit Amount\"}]"},
    {"column": "explanation_text", "type": "text", "description": "Human-readable explanation summary.", "example": "External credit reference quality increased risk."},
    {"column": "CREDIT_INCOME_RATIO", "type": "numeric", "description": "Requested credit divided by income.", "example": 2.8},
    {"column": "EXT_SOURCE_MEAN", "type": "numeric", "description": "Mean external source score when available.", "example": 0.52},
    {"column": "INST_LATE_COUNT", "type": "numeric", "description": "Late payment count when available.", "example": 1},
    {"column": "coverage_tier", "type": "text", "description": "Coverage tier such as REDUCED or FULL.", "example": "REDUCED"},
    {"column": "model_version", "type": "text", "description": "Latest model version used for scoring.", "example": "reduced_v2.1.0"},
]


def resolve_alias(term: str) -> Optional[str]:
    if not term:
        return None
    return COLUMN_ALIASES.get(term.strip().lower())


def fuzzy_resolve(query: str) -> str:
    result = "" if query is None else str(query)
    for alias in sorted(COLUMN_ALIASES.keys(), key=len, reverse=True):
        real_col = COLUMN_ALIASES[alias]
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        if re.search(pattern, result, flags=re.IGNORECASE):
            result = re.sub(pattern, real_col, result, flags=re.IGNORECASE)
    return result


def get_schema_prompt() -> str:
    lines: List[str] = ["Applicants table schema:"]
    for col in SCHEMA_COLUMNS:
        lines.append(
            f"  - {col['column']} ({col.get('type', 'text')}): "
            f"{col.get('description', '')} Example: {col.get('example')}"
        )
    return "\n".join(lines)


def get_schema_prompt_for_gemini() -> str:
    return get_schema_prompt()


def get_all_column_names() -> List[str]:
    return [c["column"] for c in SCHEMA_COLUMNS]


def is_valid_column(col: str) -> bool:
    return bool(col) and col in get_all_column_names()
