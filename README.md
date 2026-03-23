# MasterMind Credit Scoring

MasterMind is an explainable credit scoring system built on the Home Credit Default Risk dataset. Modules 1–4 are the offline pipeline: data preparation, frozen feature builders, model training/calibration, and fairness or explainability outputs. Module 5 is the Flask API and is the terminal consumer of that persisted artifact stack.

## Workflow

The current workflow is:

1. Run Module 1 to create processed splits and `data/processed/processed_artifact_manifest.json`.
2. Run Module 2 to persist the lineage-validated FULL builder.
3. Run Module 3 to persist trained models, calibrators, SHAP explainers, and `artifacts/reproducibility_report.json`.
4. Run Module 4 to persist `artifacts/model_fairness_audit_passed.joblib`.
5. Start Module 5, which eagerly validates and loads those artifacts once at startup.

Real mode is strict and fail-fast. Startup is expected to fail if processed lineage is missing, builders cannot be validated, artifacts are missing, or the stack looks incompatible. Those failures are safeguards, not bugs.

The corrected forward proxy-time regime from the repaired workflow supersedes earlier split-based metrics. Old quarantined artifacts must not be reused.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Real-Mode API Prerequisites

Module 5 requires these persisted artifacts before real-mode startup:

- `data/processed/processed_artifact_manifest.json`
- validated FULL builder loaded through `src/builder_artifacts.py`
- `artifacts/full_model.joblib`
- `artifacts/full_calibrator.joblib`
- `artifacts/full_shap_explainer.joblib`
- `artifacts/model_fairness_audit_passed.joblib`
- optional `artifacts/reproducibility_report.json` for deployed metadata

`app_test_adv.pkl` is diagnostic-only and is not used for live scoring.

## Running The API

Real mode:

```bash
set ARTIFACT_DIR=artifacts/
set DATA_PROCESSED_DIR=data/processed/
python -c "from src.api.app import create_app; app = create_app(); app.run(host='0.0.0.0', port=5000)"
```

Mock mode for local bring-up and tests:

```bash
python -c "from src.api.app import create_app; app = create_app(mock_mode=True); app.run(host='127.0.0.1', port=5000)"
```

Mock mode is for local development and tests only. It uses temp-scoped runtime locations by default and must not be treated as a production scoring path.

Frontend demo:

- Open `/` or `/demo` after the Flask app starts.
- The demo frontend is served by the same Flask app and sends real requests to `/health` and `/score`.
- It is a scratch-built UI for testing the complete backend pipeline, not a separate scoring implementation.

## Running API Tests

```bash
python -m pytest tests/test_api.py
```

The test suite is designed to run in mock mode without Kaggle raw data. Some strict real-mode startup checks are exercised with temporary fake artifacts.

## Endpoints

`GET /health`

Returns:

```json
{
  "status": "ok",
  "model_version": "full_v2.1.0|router_v1.0.0|policy_v1.0.0|fairness_v2026Q1",
  "fairness_audit_passed": true,
  "coverage_tiers_available": ["FULL"]
}
```

`POST /score`

FULL example:

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
  "previous_agg": { "...": "all required previous aggregate fields" },
  "installments_agg": { "...": "all required installments aggregate fields" },
  "pos_cash_agg": { "...": "all required POS aggregate fields" },
  "credit_card_agg": { "...": "all required credit card aggregate fields" }
}
```

Application-only starter payloads are not supported in the current API. `/score` expects the complete FULL payload.

Success response:

```json
{
  "probability_of_default": 0.21,
  "decision": "REVIEW",
  "escalate": true,
  "top_5_explanations": [
    { "feature": "BUREAU_LOAN_COUNT", "reason": "External credit history indicates elevated repayment risk" }
  ],
  "model_version": "full_v2.1.0|router_v1.0.0|policy_v1.0.0|fairness_v2026Q1",
  "calibrated": true,
  "model_fairness_audit_passed": true,
  "fairness_audit_version": "proxy_audit_2026Q1_v1.0",
  "coverage_tier": "FULL"
}
```

`model_fairness_audit_passed` reflects the offline fairness artifact produced by Module 4. The API does not rerun fairness checks during live scoring.
