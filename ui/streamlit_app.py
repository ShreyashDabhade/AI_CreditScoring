"""Production-style Streamlit UI for the credit scoring API."""

from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st

from src.api import AGG_REQUIRED_FIELDS, APPLICATION_REQUIRED_FIELDS

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:5000")
HEALTH_PATH = "/health"
SCORE_PATH = "/score"
BATCH_PATH = "/score/batch"
CATEGORICAL_FIELDS = {
    "NAME_CONTRACT_TYPE",
    "NAME_TYPE_SUITE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
}
CATEGORICAL_OPTIONS = {
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
    "NAME_TYPE_SUITE": ["Unaccompanied", "Family", "Spouse, partner"],
    "NAME_EDUCATION_TYPE": [
        "Higher education",
        "Secondary / secondary special",
        "Incomplete higher",
    ],
    "NAME_FAMILY_STATUS": ["Married", "Single / not married", "Civil marriage"],
    "OCCUPATION_TYPE": ["Laborers", "Core staff", "Sales staff"],
    "ORGANIZATION_TYPE": ["Business Entity Type 3", "Self-employed", "School"],
    "WEEKDAY_APPR_PROCESS_START": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"],
}
FIELD_DEFAULTS = {
    "AMT_INCOME_TOTAL_CAPPED": 120000.0,
    "AMT_CREDIT": 250000.0,
    "AMT_ANNUITY": 25000.0,
    "AMT_GOODS_PRICE": 220000.0,
    "DAYS_BIRTH": -12000.0,
    "DAYS_EMPLOYED": -1500.0,
    "DAYS_REGISTRATION": -3000.0,
    "DAYS_ID_PUBLISH": -2000.0,
    "DAYS_LAST_PHONE_CHANGE": -1000.0,
    "REGION_POPULATION_RELATIVE": 0.02,
    "EXT_SOURCE_1": 0.2,
    "EXT_SOURCE_2": 0.4,
    "EXT_SOURCE_3": 0.6,
    "CNT_FAM_MEMBERS": 2.0,
    "OWN_CAR_AGE": 5.0,
    "OBS_30_CNT_SOCIAL_CIRCLE": 1.0,
    "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
    "OBS_60_CNT_SOCIAL_CIRCLE": 1.0,
    "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
    "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
    "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
    "AMT_REQ_CREDIT_BUREAU_WEEK": 1.0,
    "AMT_REQ_CREDIT_BUREAU_MON": 1.0,
    "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "NAME_TYPE_SUITE": "Unaccompanied",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS": "Married",
    "OCCUPATION_TYPE": "Laborers",
    "ORGANIZATION_TYPE": "Business Entity Type 3",
    "WEEKDAY_APPR_PROCESS_START": "MONDAY",
    "DAYS_EMPLOYED_ANOM": 0.0,
    "BUREAU_LOAN_COUNT": 2.0,
    "BUREAU_ACTIVE_COUNT": 1.0,
    "BUREAU_CLOSED_COUNT": 1.0,
    "BUREAU_AMT_CREDIT_SUM_SUM": 50000.0,
    "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": 10000.0,
    "BUREAU_DEBT_TO_CREDIT_RATIO": 0.2,
    "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM": 0.0,
    "BUREAU_CREDIT_DAY_OVERDUE_MAX": 0.0,
    "BUREAU_DAYS_CREDIT_MAX": -200.0,
    "BUREAU_CNT_CREDIT_PROLONG_SUM": 0.0,
    "PREV_APP_COUNT": 3.0,
    "PREV_APPROVED_COUNT": 2.0,
    "PREV_REFUSED_COUNT": 1.0,
    "PREV_APPROVAL_RATE": 0.67,
    "PREV_REFUSAL_RATE": 0.33,
    "PREV_AMT_APPLICATION_MEAN": 210000.0,
    "PREV_AMT_CREDIT_MEAN": 195000.0,
    "PREV_AMT_GOODS_PRICE_MEAN": 187000.0,
    "PREV_APP_CREDIT_DIFF_MEAN": 15000.0,
    "PREV_DAYS_DECISION_MAX": -120.0,
    "PREV_RATE_DOWN_PAYMENT_MEAN": 0.08,
    "INST_RECORD_COUNT": 12.0,
    "INST_MISSED_RATE": 0.08,
    "INST_DPD_MEAN": 4.0,
    "INST_DPD_MAX": 12.0,
    "INST_PAYMENT_RATIO_MEAN": 0.95,
    "INST_PAYMENT_RATIO_MIN": 0.72,
    "INST_LATE_COUNT": 3.0,
    "POS_RECORD_COUNT": 6.0,
    "POS_DPD_MEAN": 1.5,
    "POS_DPD_MAX": 7.0,
    "POS_DPD_DEF_MEAN": 0.5,
    "POS_DPD_DEF_MAX": 4.0,
    "POS_COMPLETED_RATE": 0.6,
    "POS_ACTIVE_RATE": 0.4,
    "POS_CNT_INSTALMENT_FUTURE_MEAN": 2.0,
    "CC_RECORD_COUNT": 8.0,
    "CC_BALANCE_MEAN": 18000.0,
    "CC_LIMIT_MEAN": 60000.0,
    "CC_UTILIZATION_MEAN": 0.3,
    "CC_PAYMENT_RATIO_MEAN": 1.1,
    "CC_DPD_MEAN": 1.0,
    "CC_DPD_MAX": 6.0,
    "CC_DRAWINGS_ATM_SUM": 4500.0,
    "CC_DRAWINGS_CURRENT_SUM": 9000.0,
}


def api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


def load_health() -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(api_url(HEALTH_PATH), timeout=5)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)


def submit_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.post(api_url(SCORE_PATH), json=payload, timeout=30)
        body = response.json()
        if response.ok:
            return body, None
        return None, body.get("message", "Scoring failed.")
    except Exception as exc:
        return None, str(exc)


def submit_batch(payloads: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.post(api_url(BATCH_PATH), json=payloads, timeout=60)
        body = response.json()
        if response.ok:
            return body, None
        return None, body.get("message", "Batch scoring failed.")
    except Exception as exc:
        return None, str(exc)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f9fc; color: #10273a; }
        .main-title { font-size: 2.2rem; font-weight: 700; color: #0072CE; margin-bottom: 0.25rem; }
        .subtitle { color: #5f6f82; margin-bottom: 1.5rem; }
        .card {
            background: white;
            border-radius: 18px;
            padding: 1.25rem;
            border: 1px solid #dde5ef;
            box-shadow: 0 8px 28px rgba(17, 39, 58, 0.06);
            margin-bottom: 1rem;
        }
        .decision-approve { border-left: 6px solid #1f9d55; }
        .decision-review { border-left: 6px solid #dd7a11; }
        .decision-decline { border-left: 6px solid #cf3f3f; }
        .metric-label { color: #5f6f82; font-size: 0.9rem; }
        .metric-value { font-size: 1.5rem; font-weight: 700; color: #0b2f53; }
        .reason-chip {
            display: inline-block;
            background: #eef4fb;
            border: 1px solid #d7e4f3;
            color: #20476d;
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            margin: 0.2rem 0.3rem 0.2rem 0;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_health_panel(health: dict[str, Any] | None, error: str | None) -> None:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("API Status")
        if error:
            st.error(f"Backend unavailable: {error}")
        elif health is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Status", health.get("status", "unknown").upper())
            col2.metric("Model Version", health.get("model_version", "n/a"))
            col3.metric(
                "Fairness Audit",
                "Passed" if health.get("fairness_audit_passed") else "Pending",
            )
            tiers = ", ".join(health.get("coverage_tiers_available", []))
            st.caption(f"Available tiers: {tiers or 'Unavailable'}")
        st.markdown("</div>", unsafe_allow_html=True)


def render_field(field_name: str, section_name: str) -> Any:
    key = f"{section_name}.{field_name}"
    if field_name in CATEGORICAL_FIELDS:
        options = CATEGORICAL_OPTIONS[field_name]
        default_value = FIELD_DEFAULTS[field_name]
        default_index = options.index(default_value) if default_value in options else 0
        return st.selectbox(field_name, options, index=default_index, key=key)
    return st.number_input(
        field_name,
        value=float(FIELD_DEFAULTS.get(field_name, 0.0)),
        step=1.0,
        format="%.6f",
        key=key,
    )


def build_payload(coverage_tier: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"application": {}}
    for field_name in APPLICATION_REQUIRED_FIELDS:
        payload["application"][field_name] = st.session_state[f"application.{field_name}"]
    if coverage_tier == "FULL":
        for section_name, field_names in AGG_REQUIRED_FIELDS.items():
            payload[section_name] = {}
            for field_name in field_names:
                payload[section_name][field_name] = st.session_state[f"{section_name}.{field_name}"]
    return payload


def render_single_scoring_tab() -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Single Application Scoring")
    coverage_tier = st.radio("Coverage Tier", ["REDUCED", "FULL"], horizontal=True)
    with st.form("single_score_form"):
        st.markdown("#### Applicant Information")
        app_cols = st.columns(2)
        for index, field_name in enumerate(APPLICATION_REQUIRED_FIELDS):
            with app_cols[index % 2]:
                render_field(field_name, "application")

        if coverage_tier == "FULL":
            st.markdown("#### Aggregate Inputs")
            for section_name, field_names in AGG_REQUIRED_FIELDS.items():
                with st.expander(section_name.replace("_", " ").title(), expanded=False):
                    cols = st.columns(2)
                    for index, field_name in enumerate(field_names):
                        with cols[index % 2]:
                            render_field(field_name, section_name)

        submitted = st.form_submit_button("Run Credit Score")

    if submitted:
        payload = build_payload(coverage_tier)
        response, error = submit_payload(payload)
        if error:
            st.error(error)
        elif response is not None:
            render_result(response)
            with st.expander("Request Payload", expanded=False):
                st.json(payload)
    st.markdown("</div>", unsafe_allow_html=True)


def render_result(response: dict[str, Any]) -> None:
    decision = response.get("decision", "REVIEW").upper()
    decision_class = {
        "APPROVE": "decision-approve",
        "REVIEW": "decision-review",
        "DECLINE": "decision-decline",
    }.get(decision, "decision-review")
    st.markdown(f'<div class="card {decision_class}">', unsafe_allow_html=True)
    st.subheader("Decision")
    col1, col2, col3 = st.columns(3)
    col1.metric("Probability of Default", f"{response.get('probability_of_default', 0.0):.4f}")
    col2.metric("Decision", decision)
    col3.metric("Coverage Tier", response.get("coverage_tier", "n/a"))
    st.caption(f"Model Version: {response.get('model_version', 'n/a')}")
    st.markdown("#### Top 5 Explanations")
    for item in response.get("top_5_explanations", []):
        reason = f"{item.get('feature', 'feature')}: {item.get('reason', '')}"
        st.markdown(f'<span class="reason-chip">{reason}</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_batch_tab() -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Batch Scoring")
    st.caption("Paste a JSON array of request payloads that match the Flask API contract.")
    default_batch = json.dumps(
        [
            {
                "application": {
                    field_name: FIELD_DEFAULTS[field_name]
                    for field_name in APPLICATION_REQUIRED_FIELDS
                }
            }
        ],
        indent=2,
    )
    raw_json = st.text_area("Batch Payload", value=default_batch, height=320)
    if st.button("Run Batch Score"):
        try:
            payloads = json.loads(raw_json)
            if not isinstance(payloads, list):
                st.error("Batch payload must be a JSON array.")
            else:
                response, error = submit_batch(payloads)
                if error:
                    st.error(error)
                elif response is not None:
                    st.success("Batch scoring completed.")
                    st.json(response)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="AI Credit Scoring", layout="wide")
    inject_styles()
    st.markdown('<div class="main-title">AI Credit Scoring Console</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Production-style underwriting interface powered by the Flask scoring API.</div>',
        unsafe_allow_html=True,
    )

    health, health_error = load_health()
    render_health_panel(health, health_error)

    tab_single, tab_batch = st.tabs(["Single Score", "Batch Score"])
    with tab_single:
        render_single_scoring_tab()
    with tab_batch:
        render_batch_tab()


if __name__ == "__main__":
    main()
