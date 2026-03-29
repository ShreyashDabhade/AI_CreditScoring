# Backend API Contract

This document freezes the backend/frontend contract for the current backend branch.

## Scope

- Runtime processed path default: `data/processed/`
- Runtime artifact path default: `artifacts/`
- Real API startup path: `src.api.app.create_app(mock_mode=False, strict_artifacts=True)`
- Supported split modes in config: `proxy_time`, `random_stratified`
- Current default split mode: `proxy_time`

Offline experiment folders and reports are not API inputs.

## Canonical Runtime Artifacts

The real API validates and loads only these runtime inputs:

- `data/processed/processed_artifact_manifest.json`
- `artifacts/full_feature_builder.joblib`
- `artifacts/full_feature_builder.manifest.json`
- `artifacts/reduced_feature_builder.joblib`
- `artifacts/reduced_feature_builder.manifest.json`
- `artifacts/full_model.joblib`
- `artifacts/full_calibrator.joblib`
- `artifacts/full_shap_explainer.joblib`
- `artifacts/reduced_model.joblib`
- `artifacts/reduced_calibrator.joblib`
- `artifacts/reduced_shap_explainer.joblib`
- `artifacts/model_fairness_audit_passed.joblib`

Optional metadata:

- `artifacts/reproducibility_report.json`

The API does not load nested experiment folders such as `artifacts/reduced_thin_blend/` or `artifacts/random_stratified_reduced/`.

## GET /health

Response shape:

```json
{
  "status": "ok",
  "model_version": "full_weighted_blend_v2.2.0",
  "fairness_audit_passed": false,
  "coverage_tiers_available": ["FULL", "REDUCED"]
}
```

Field meanings:

- `status`: fixed string `ok` when startup succeeded
- `model_version`: runtime health version for the deployed FULL stack
- `fairness_audit_passed`: boolean loaded from `artifacts/model_fairness_audit_passed.joblib`
- `coverage_tiers_available`: currently `["FULL", "REDUCED"]`

## POST /score

### Accepted Top-Level Shapes

REDUCED request:

```json
{
  "application": { "...": "..." }
}
```

FULL request:

```json
{
  "application": { "...": "..." },
  "bureau_agg": { "...": "..." },
  "previous_agg": { "...": "..." },
  "installments_agg": { "...": "..." },
  "pos_cash_agg": { "...": "..." },
  "credit_card_agg": { "...": "..." }
}
```

Partial FULL payloads are rejected.

### Exact Required Application Fields

```json
[
  "AMT_INCOME_TOTAL_CAPPED",
  "AMT_CREDIT",
  "AMT_ANNUITY",
  "AMT_GOODS_PRICE",
  "DAYS_BIRTH",
  "DAYS_EMPLOYED",
  "DAYS_REGISTRATION",
  "DAYS_ID_PUBLISH",
  "DAYS_LAST_PHONE_CHANGE",
  "REGION_POPULATION_RELATIVE",
  "EXT_SOURCE_1",
  "EXT_SOURCE_2",
  "EXT_SOURCE_3",
  "CNT_FAM_MEMBERS",
  "OWN_CAR_AGE",
  "OBS_30_CNT_SOCIAL_CIRCLE",
  "DEF_30_CNT_SOCIAL_CIRCLE",
  "OBS_60_CNT_SOCIAL_CIRCLE",
  "DEF_60_CNT_SOCIAL_CIRCLE",
  "AMT_REQ_CREDIT_BUREAU_HOUR",
  "AMT_REQ_CREDIT_BUREAU_DAY",
  "AMT_REQ_CREDIT_BUREAU_WEEK",
  "AMT_REQ_CREDIT_BUREAU_MON",
  "AMT_REQ_CREDIT_BUREAU_QRT",
  "AMT_REQ_CREDIT_BUREAU_YEAR",
  "NAME_CONTRACT_TYPE",
  "NAME_TYPE_SUITE",
  "NAME_EDUCATION_TYPE",
  "NAME_FAMILY_STATUS",
  "OCCUPATION_TYPE",
  "ORGANIZATION_TYPE",
  "WEEKDAY_APPR_PROCESS_START",
  "DAYS_EMPLOYED_ANOM"
]
```

`CODE_GENDER` is explicitly forbidden.

### Exact Required FULL Aggregate Fields

`bureau_agg`

```json
[
  "BUREAU_LOAN_COUNT",
  "BUREAU_ACTIVE_COUNT",
  "BUREAU_CLOSED_COUNT",
  "BUREAU_AMT_CREDIT_SUM_SUM",
  "BUREAU_AMT_CREDIT_SUM_DEBT_SUM",
  "BUREAU_DEBT_TO_CREDIT_RATIO",
  "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM",
  "BUREAU_CREDIT_DAY_OVERDUE_MAX",
  "BUREAU_DAYS_CREDIT_MAX",
  "BUREAU_CNT_CREDIT_PROLONG_SUM"
]
```

`previous_agg`

```json
[
  "PREV_APP_COUNT",
  "PREV_APPROVED_COUNT",
  "PREV_REFUSED_COUNT",
  "PREV_APPROVAL_RATE",
  "PREV_REFUSAL_RATE",
  "PREV_AMT_APPLICATION_MEAN",
  "PREV_AMT_CREDIT_MEAN",
  "PREV_AMT_GOODS_PRICE_MEAN",
  "PREV_APP_CREDIT_DIFF_MEAN",
  "PREV_DAYS_DECISION_MAX",
  "PREV_RATE_DOWN_PAYMENT_MEAN"
]
```

`installments_agg`

```json
[
  "INST_RECORD_COUNT",
  "INST_MISSED_RATE",
  "INST_DPD_MEAN",
  "INST_DPD_MAX",
  "INST_PAYMENT_RATIO_MEAN",
  "INST_PAYMENT_RATIO_MIN",
  "INST_LATE_COUNT"
]
```

`pos_cash_agg`

```json
[
  "POS_RECORD_COUNT",
  "POS_DPD_MEAN",
  "POS_DPD_MAX",
  "POS_DPD_DEF_MEAN",
  "POS_DPD_DEF_MAX",
  "POS_COMPLETED_RATE",
  "POS_ACTIVE_RATE",
  "POS_CNT_INSTALMENT_FUTURE_MEAN"
]
```

`credit_card_agg`

```json
[
  "CC_RECORD_COUNT",
  "CC_BALANCE_MEAN",
  "CC_LIMIT_MEAN",
  "CC_UTILIZATION_MEAN",
  "CC_PAYMENT_RATIO_MEAN",
  "CC_DPD_MEAN",
  "CC_DPD_MAX",
  "CC_DRAWINGS_ATM_SUM",
  "CC_DRAWINGS_CURRENT_SUM"
]
```

### Example REDUCED Request

```json
{
  "application": {
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
    "DAYS_EMPLOYED_ANOM": 0
  }
}
```

### Example FULL Request

```json
{
  "application": {
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
    "DAYS_EMPLOYED_ANOM": 0
  },
  "bureau_agg": {
    "BUREAU_LOAN_COUNT": 2.0,
    "BUREAU_ACTIVE_COUNT": 1.0,
    "BUREAU_CLOSED_COUNT": 1.0,
    "BUREAU_AMT_CREDIT_SUM_SUM": 50000.0,
    "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": 10000.0,
    "BUREAU_DEBT_TO_CREDIT_RATIO": 0.2,
    "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM": 0.0,
    "BUREAU_CREDIT_DAY_OVERDUE_MAX": 0.0,
    "BUREAU_DAYS_CREDIT_MAX": -200.0,
    "BUREAU_CNT_CREDIT_PROLONG_SUM": 0.0
  },
  "previous_agg": {
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
    "PREV_RATE_DOWN_PAYMENT_MEAN": 0.08
  },
  "installments_agg": {
    "INST_RECORD_COUNT": 12.0,
    "INST_MISSED_RATE": 0.08,
    "INST_DPD_MEAN": 4.0,
    "INST_DPD_MAX": 12.0,
    "INST_PAYMENT_RATIO_MEAN": 0.95,
    "INST_PAYMENT_RATIO_MIN": 0.72,
    "INST_LATE_COUNT": 3.0
  },
  "pos_cash_agg": {
    "POS_RECORD_COUNT": 6.0,
    "POS_DPD_MEAN": 1.5,
    "POS_DPD_MAX": 7.0,
    "POS_DPD_DEF_MEAN": 0.5,
    "POS_DPD_DEF_MAX": 4.0,
    "POS_COMPLETED_RATE": 0.6,
    "POS_ACTIVE_RATE": 0.4,
    "POS_CNT_INSTALMENT_FUTURE_MEAN": 2.0
  },
  "credit_card_agg": {
    "CC_RECORD_COUNT": 8.0,
    "CC_BALANCE_MEAN": 18000.0,
    "CC_LIMIT_MEAN": 60000.0,
    "CC_UTILIZATION_MEAN": 0.3,
    "CC_PAYMENT_RATIO_MEAN": 1.1,
    "CC_DPD_MEAN": 1.0,
    "CC_DPD_MAX": 6.0,
    "CC_DRAWINGS_ATM_SUM": 4500.0,
    "CC_DRAWINGS_CURRENT_SUM": 9000.0
  }
}
```

### Success Response Shape

```json
{
  "probability_of_default": 0.06660137120470128,
  "decision": "APPROVE",
  "escalate": false,
  "top_5_explanations": [
    {
      "feature": "EXT_SOURCE_MEAN",
      "reason": "Combined application and repayment profile increased model risk"
    },
    {
      "feature": "CREDIT_TERM_RATIO",
      "reason": "Combined application and repayment profile increased model risk"
    },
    {
      "feature": "OWN_CAR_AGE",
      "reason": "Combined application and repayment profile increased model risk"
    },
    {
      "feature": "INST_MISSED_RATE",
      "reason": "Combined application and repayment profile increased model risk"
    },
    {
      "feature": "PREV_APP_CREDIT_DIFF_MEAN",
      "reason": "Combined application and repayment profile increased model risk"
    }
  ],
  "model_version": "full_weighted_blend_v2.2.0",
  "calibrated": true,
  "model_fairness_audit_passed": false,
  "fairness_audit_version": "proxy_audit_2026Q1_v1.1",
  "coverage_tier": "FULL"
}
```

Notes:

- `top_5_explanations` always contains 5 items on success
- `coverage_tier` is either `REDUCED` or `FULL`
- `model_version` is tier-specific
- `decision` is one of `APPROVE`, `REVIEW`, `DECLINE`
- `escalate` is `true` only when `decision` is `REVIEW`

### Error Responses

Malformed JSON or invalid envelope:

```json
{
  "error_code": "bad_request",
  "message": "Bad request."
}
```

Contract validation error:

```json
{
  "error_code": "missing_application_fields",
  "message": "Application payload is missing required fields.",
  "missing_fields": ["AMT_CREDIT"]
}
```

Known validation error codes:

- `bad_request`
- `missing_application`
- `forbidden_field_code_gender`
- `partial_full_payload_not_allowed`
- `missing_application_fields`
- `missing_aggregate_fields`
- `starter_not_supported_in_mvp`
- `internal_error`
