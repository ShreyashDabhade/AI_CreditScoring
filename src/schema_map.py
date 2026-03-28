"""Schema map single-source-of-truth for the live ``applicants`` table.

This module maps natural-language aliases to the actual SQLite column names
used by the chatbot database and exposes helpers that provide schema context
for validation and LLM prompts.

Pure Python, no external dependencies.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

#----------------------------------------------------------------------
# Aliases: natural language term (lowercase) -> exact DB column name
#----------------------------------------------------------------------
COLUMN_ALIASES: Dict[str, str] = {
    # income aliases
    "income": "AMT_INCOME_TOTAL_CAPPED",
    "salary": "AMT_INCOME_TOTAL_CAPPED",
    "earnings": "AMT_INCOME_TOTAL_CAPPED",
    "annual income": "AMT_INCOME_TOTAL_CAPPED",
    "yearly income": "AMT_INCOME_TOTAL_CAPPED",

    # identifiers
    "id": "SK_ID_CURR",
    "applicant id": "SK_ID_CURR",
    "application id": "SK_ID_CURR",
    "application no": "SK_ID_CURR",
    "app id": "SK_ID_CURR",
    "customer id": "SK_ID_CURR",
    "client id": "SK_ID_CURR",
    "loan id": "SK_ID_CURR",
    "applicant_id": "SK_ID_CURR",
    "sk_id": "SK_ID_CURR",
    "sk id": "SK_ID_CURR",
    "sk_id_curr": "SK_ID_CURR",
    "sk id curr": "SK_ID_CURR",

    # loan / credit
    "loan amount": "AMT_CREDIT",
    "credit amount": "AMT_CREDIT",
    "loan value": "AMT_CREDIT",

    # scores and predictions
    "score": "credit_score",
    "credit score": "credit_score",
    "risk": "risk_band",
    "risk band": "risk_band",
    "risk level": "risk_band",
    "risk score": "credit_score",
    "band": "risk_band",
    "category": "risk_band",
    "prediction": "prediction",
    "decision": "prediction",
    "result": "prediction",
    "status": "prediction",
    "outcome": "prediction",
    "approved": "prediction",
    "rejected": "prediction",
    "probability": "probability",
    "default probability": "probability",
    "default risk": "probability",
    "default chance": "probability",

    # explanations / features
    "explanation": "explanation_text",
    "reason": "explanation_text",
    "top features": "top_features",
    "shap": "top_features",

    # financial details
    "annuity": "AMT_ANNUITY",
    "employed years": "DAYS_EMPLOYED",
    "employment": "DAYS_EMPLOYED",
    "years employed": "DAYS_EMPLOYED",
    "job duration": "DAYS_EMPLOYED",

    # demographics
    "age": "AGE_YEARS",
    "gender": "CODE_GENDER",
    "sex": "CODE_GENDER",
    "education": "NAME_EDUCATION_TYPE",
    "edu": "NAME_EDUCATION_TYPE",
    "qualification": "NAME_EDUCATION_TYPE",
    "family": "CNT_FAM_MEMBERS",
    "family members": "CNT_FAM_MEMBERS",
    "dependents": "CNT_FAM_MEMBERS",
}

# Exact alias resolution is kept backward-compatible for the current parser and
# legacy SQLite fixtures. Fuzzy prompt rewriting uses ``COLUMN_ALIASES``
# directly, so Gemini still receives the richer production-style schema names.
_EXACT_ALIAS_RESOLUTION_OVERRIDES: Dict[str, str] = {
    "income": "income",
    "salary": "income",
    "earnings": "income",
    "annual income": "income",
    "yearly income": "income",
    "id": "applicant_id",
    "applicant id": "applicant_id",
    "application id": "applicant_id",
    "application no": "applicant_id",
    "app id": "applicant_id",
    "customer id": "applicant_id",
    "client id": "applicant_id",
    "loan id": "applicant_id",
    "applicant_id": "applicant_id",
    "sk_id": "applicant_id",
    "sk id": "applicant_id",
    "sk_id_curr": "applicant_id",
    "sk id curr": "applicant_id",
    "loan amount": "mock_credit",
    "credit amount": "mock_credit",
    "loan value": "mock_credit",
    "annuity": "mock_annuity",
    "age": "age",
}


#----------------------------------------------------------------------
# SCHEMA_COLUMNS: list of dicts describing columns for prompts and validation
#----------------------------------------------------------------------
SCHEMA_COLUMNS: List[Dict[str, str]] = [
    {"column": "SK_ID_CURR", "type": "numeric", "description": "Unique applicant identifier.", "example": 99999},
    {"column": "AMT_INCOME_TOTAL_CAPPED", "type": "numeric", "description": "Applicant annual income after capping extreme values.", "example": 750000.0},
    {"column": "AMT_CREDIT", "type": "numeric", "description": "Requested loan amount.", "example": 250000.0},
    {"column": "AMT_ANNUITY", "type": "numeric", "description": "Loan annuity payment amount.", "example": 25000.0},
    {"column": "AGE_YEARS", "type": "numeric", "description": "Applicant age in years.", "example": 34},
    {"column": "CODE_GENDER", "type": "text", "description": "Applicant gender code.", "example": "M"},
    {"column": "NAME_EDUCATION_TYPE", "type": "text", "description": "Applicant education level.", "example": "Higher education"},
    {"column": "CNT_FAM_MEMBERS", "type": "numeric", "description": "Number of family members associated with the application.", "example": 3},
    {"column": "DAYS_EMPLOYED", "type": "numeric", "description": "Employment duration in days (often negative in the source dataset).", "example": -1460},
    {"column": "prediction", "type": "text", "description": "Final model decision such as APPROVE or REJECT.", "example": "REJECT"},
    {"column": "credit_score", "type": "numeric", "description": "Model-derived credit score for the applicant.", "example": 720},
    {"column": "risk_band", "type": "categorical", "description": "Risk label such as LOW, MEDIUM, or HIGH.", "example": "LOW"},
    {"column": "probability", "type": "numeric", "description": "Predicted probability of default between 0 and 1.", "example": 0.1234},
    {"column": "top_features", "type": "text", "description": "Serialized top contributing features for the prediction.", "example": "[{\"feature\": \"EXT_SOURCE_1\"}]"},
    {"column": "explanation_text", "type": "text", "description": "Human-readable explanation for the decision.", "example": "Top drivers identified"},
    {"column": "applicant_id", "type": "numeric", "description": "Unique applicant identifier.", "example": 99999},
    {"column": "income", "type": "numeric", "description": "Applicant income stored in the chatbot database.", "example": 750000.0},
    {"column": "age", "type": "numeric", "description": "Applicant age in years.", "example": 34},
    {"column": "mock_income", "type": "numeric", "description": "Derived mock feature available in mock-mode data.", "example": 0.5},
    {"column": "mock_credit", "type": "numeric", "description": "Derived mock credit feature available in mock-mode data.", "example": 250000.0},
    {"column": "mock_annuity", "type": "numeric", "description": "Derived mock annuity feature available in mock-mode data.", "example": 25000.0},
    {"column": "mock_ratio", "type": "numeric", "description": "Derived mock ratio feature available in mock-mode data.", "example": 0.33},
    {"column": "mock_ext_mean", "type": "numeric", "description": "Derived mock external-score average.", "example": 0.42},
    {"column": "mock_social", "type": "numeric", "description": "Derived mock social-circle feature.", "example": 2.0},
    {"column": "mock_bureau", "type": "numeric", "description": "Derived mock bureau feature.", "example": 1.0},
    {"column": "mock_previous", "type": "numeric", "description": "Derived mock previous-application feature.", "example": 3.0},
    {"column": "mock_installments", "type": "numeric", "description": "Derived mock installment feature.", "example": 12.0},
    {"column": "mock_pos", "type": "numeric", "description": "Derived mock POS feature.", "example": 6.0},
    {"column": "mock_cc", "type": "numeric", "description": "Derived mock credit-card feature.", "example": 8.0},
]


def resolve_alias(term: str) -> Optional[str]:
    """Resolve a natural language term to an exact DB column name.

    Returns the column name if an alias is known, otherwise None.
    """
    if not term:
        return None
    key = term.strip().lower()
    if key in _EXACT_ALIAS_RESOLUTION_OVERRIDES:
        return _EXACT_ALIAS_RESOLUTION_OVERRIDES[key]
    return COLUMN_ALIASES.get(key)


def fuzzy_resolve(query: str) -> str:
    """Replace any alias substrings in a raw user query with schema column names.

    This is used for Gemini prompt construction so the model sees the
    production-style schema even when users phrase requests informally.
    """
    result = "" if query is None else str(query)
    sorted_aliases = sorted(COLUMN_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        real_col = COLUMN_ALIASES[alias]
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        if re.search(pattern, result, flags=re.IGNORECASE) and real_col.lower() not in result.lower():
            result = re.sub(pattern, real_col, result, flags=re.IGNORECASE)
    return result


def get_schema_prompt() -> str:
    """Produce a formatted schema listing suitable for LLM prompts.

    Each column is shown on its own line in the format:
      - COLUMN_NAME (type): description Example: <example>
    """
    lines: List[str] = [
        "Applicants table schema:",
    ]
    for col in SCHEMA_COLUMNS:
        column = col.get("column")
        ctype = col.get("type", "text")
        desc = col.get("description", "")
        example = col.get("example")
        lines.append(f"  - {column} ({ctype}): {desc} Example: {example}")
    return "\n".join(lines)


def get_schema_prompt_for_gemini() -> str:
    """Return a richer grouped schema description for Gemini SQL prompts."""
    lines: List[str] = []
    lines.append("Applicants table schema:")
    lines.append("TABLE: applicants")
    lines.append("=" * 50)
    lines.append("EXACT column names you MUST use in SQL:")
    lines.append("")

    groups = {
        "IDENTITY": [
            "SK_ID_CURR       (numeric) - applicant ID. User may say: id, sk_id, application id, customer id",
        ],
        "FINANCIAL": [
            "AMT_INCOME_TOTAL_CAPPED (numeric) - annual income. User may say: income, salary, earnings",
            "AMT_CREDIT       (numeric) - loan amount requested. User may say: loan amount, credit amount",
            "AMT_ANNUITY      (numeric) - loan annuity payment",
        ],
        "DECISION": [
            "prediction       (text)    - APPROVE or REJECT. User may say: decision, status, outcome, result",
            "credit_score     (numeric) - score 300-850. User may say: score, risk score",
            "risk_band        (text)    - LOW / MEDIUM / HIGH. User may say: risk, band, category",
            "probability      (numeric) - default probability 0.0-1.0. User may say: default risk, chance",
        ],
        "PERSONAL": [
            "AGE_YEARS        (numeric) - age in years. User may say: age",
            "CODE_GENDER      (text)    - M or F. User may say: gender, sex",
            "NAME_EDUCATION_TYPE (text) - education level. User may say: education, qualification",
            "CNT_FAM_MEMBERS  (numeric) - family size. User may say: family, dependents",
            "DAYS_EMPLOYED    (numeric) - employment duration in days. User may say: employment, job duration",
        ],
        "EXPLANATION": [
            "explanation_text (text)    - full narrative explanation of the decision",
            "top_features     (text)    - SHAP feature importances as JSON string",
        ],
    }

    for group, cols in groups.items():
        lines.append(f"[{group}]")
        for col in cols:
            lines.append(f"  - {col}")
        lines.append("")

    lines.append("CRITICAL: Never invent column names. Only use names listed above.")
    lines.append("CRITICAL: Table name is always 'applicants'. No other tables exist.")
    return "\n".join(lines)


def get_all_column_names() -> List[str]:
    """Return a list of all exact column names known in the schema."""
    return [c["column"] for c in SCHEMA_COLUMNS]


def is_valid_column(col: str) -> bool:
    """Return True if `col` is an exact column name in the schema."""
    if not col:
        return False
    return col in get_all_column_names()


if __name__ == "__main__":
    print(get_schema_prompt())
    print()
    print(get_schema_prompt_for_gemini())
    print()
    print(fuzzy_resolve("Get applicant SK_ID = 99999"))
    print(fuzzy_resolve("income > 50000"))
