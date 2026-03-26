**Real Pipeline Runbook**

Run everything from the repo root:

```powershell
cd c:\Users\shrey\Everything\AI_CreditScoring
```

This is the real-data flow. It requires these raw files in `data\raw\`:

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`
- `credit_card_balance.csv`

I’d use `venv\Scripts\python.exe` directly so you don’t get blocked by PowerShell activation policy.

---

**1. One-Time Setup**

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set the repo paths for this terminal session:

```powershell
$env:DATA_RAW_DIR = "data/raw/"
$env:DATA_PROCESSED_DIR = "data/processed/"
$env:ARTIFACT_DIR = "artifacts/"
$env:EDA_PLOTS_DIR = "notebooks/eda_plots/"
$env:SHAP_PLOTS_DIR = "notebooks/shap_plots/"
$env:FAIRNESS_PLOTS_DIR = "notebooks/fairness_plots/"
$env:EVAL_PLOTS_DIR = "notebooks/eval_plots/"
```

---

**2. Run The Offline Pipeline**

**Module 1: data processing**

```powershell
venv\Scripts\python.exe -m src.data_pipeline
```

Check the outputs:

```powershell
Get-ChildItem data\processed
Get-Content data\processed\processed_artifact_manifest.json | Select-Object -First 40
Get-Item data\data_quality_report.json
Get-ChildItem notebooks\eda_plots
```

You should see at least:

- `train.pkl`
- `val_model.pkl`
- `val_policy.pkl`
- `test.pkl`
- `app_test_adv.pkl`
- `income_cap.joblib`
- `processed_artifact_manifest.json`

**Module 2: feature builders**

```powershell
venv\Scripts\python.exe -m src.feature_engineering
```

Check that both builders load cleanly:

```powershell
@'
from src.builder_artifacts import load_validated_builders
builders = load_validated_builders(
    artifact_dir="artifacts",
    processed_dir="data/processed",
    strict_artifacts=True,
)
print("FULL", builders["FULL"].tier, len(builders["FULL"].encoded_columns_))
print("REDUCED", builders["REDUCED"].tier, len(builders["REDUCED"].encoded_columns_))
'@ | venv\Scripts\python.exe -
```

**Module 3: training + reproducibility**

```powershell
venv\Scripts\python.exe -m src.models.train
```

Check the training outputs:

```powershell
Get-ChildItem artifacts
@'
import json
with open("artifacts/reproducibility_report.json", "r", encoding="utf-8") as f:
    r = json.load(f)
print("full_model_version:", r["full_model_version"])
print("full_model_family:", r["tiers"]["FULL"]["model_family"])
print("full_selected_candidate:", r["tiers"]["FULL"]["selection"]["candidate"])
print("reduced_model_version:", r["reduced_model_version"])
print("processed_manifest_fingerprint:", r["processed_manifest_fingerprint"])
'@ | venv\Scripts\python.exe -
```

Expected current truth:

- FULL version: `full_weighted_blend_v2.2.0`
- FULL family: `weighted_blend`
- FULL selected candidate: `weighted_blend_full`
- REDUCED version: `reduced_v2.1.0`

**Module 4: fairness audit + explainability/eval plots**

```powershell
venv\Scripts\python.exe -m src.fairness_audit
```

Check the outputs:

```powershell
@'
import joblib
print("fairness_audit_passed:", joblib.load("artifacts/model_fairness_audit_passed.joblib"))
'@ | venv\Scripts\python.exe -

Get-ChildItem notebooks\fairness_plots
Get-ChildItem notebooks\shap_plots
Get-ChildItem notebooks\eval_plots
```

Expected current state in this repo: fairness still ends up `False`.

---

**3. Start The API And Score Requests**

Open a second PowerShell window in the repo root and set the same env vars:

```powershell
cd c:\Users\shrey\Everything\AI_CreditScoring
$env:DATA_RAW_DIR = "data/raw/"
$env:DATA_PROCESSED_DIR = "data/processed/"
$env:ARTIFACT_DIR = "artifacts/"
$env:EDA_PLOTS_DIR = "notebooks/eda_plots/"
$env:SHAP_PLOTS_DIR = "notebooks/shap_plots/"
$env:FAIRNESS_PLOTS_DIR = "notebooks/fairness_plots/"
$env:EVAL_PLOTS_DIR = "notebooks/eval_plots/"
```

Start the real API:

```powershell
venv\Scripts\python.exe -c "from src.api.app import create_app; app = create_app(artifact_dir='artifacts', processed_dir='data/processed', mock_mode=False, strict_artifacts=True); app.run(host='127.0.0.1', port=5000)"
```

In a third terminal, check health:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/health -Method Get
```

Generate one valid REDUCED payload and one valid FULL payload:

```powershell
@'
import json
from src.api import AGG_REQUIRED_FIELDS

application = {
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

reduced_payload = {"application": application}
full_payload = {"application": application}
for section_name, fields in AGG_REQUIRED_FIELDS.items():
    full_payload[section_name] = {
        field: float(index + 1) for index, field in enumerate(fields)
    }

with open("reduced_payload.json", "w", encoding="utf-8") as f:
    json.dump(reduced_payload, f, indent=2)

with open("full_payload.json", "w", encoding="utf-8") as f:
    json.dump(full_payload, f, indent=2)

print("Wrote reduced_payload.json and full_payload.json")
'@ | venv\Scripts\python.exe -
```

Score REDUCED:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/score -Method Post -ContentType "application/json" -InFile reduced_payload.json
```

Score FULL:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/score -Method Post -ContentType "application/json" -InFile full_payload.json
```

What you should see:

- `/health` returns `status: ok`
- `/health.model_version` is `full_weighted_blend_v2.2.0`
- REDUCED `/score` returns `coverage_tier: REDUCED`
- FULL `/score` returns `coverage_tier: FULL`
- REDUCED version is `reduced_v2.1.0`
- FULL version is `full_weighted_blend_v2.2.0`

Stop the API with `Ctrl+C`.

---

**4. Full Test Commands**

Run the full suite:

```powershell
venv\Scripts\python.exe -m pytest tests -q
```

Useful targeted runs:

```powershell
venv\Scripts\python.exe -m pytest tests/test_api.py -q
venv\Scripts\python.exe -m pytest tests/test_m2_m4_integration.py -q
venv\Scripts\python.exe -m pytest tests/test_m3_blending.py -q
venv\Scripts\python.exe -m pytest tests/test_subgroup_calibration.py -q
```

---

**5. Minimal End-To-End Sequence**

If you just want the shortest real pipeline command chain:

```powershell
cd c:\Users\shrey\Everything\AI_CreditScoring
$env:DATA_RAW_DIR = "data/raw/"
$env:DATA_PROCESSED_DIR = "data/processed/"
$env:ARTIFACT_DIR = "artifacts/"
$env:EDA_PLOTS_DIR = "notebooks/eda_plots/"
$env:SHAP_PLOTS_DIR = "notebooks/shap_plots/"
$env:FAIRNESS_PLOTS_DIR = "notebooks/fairness_plots/"
$env:EVAL_PLOTS_DIR = "notebooks/eval_plots/"

venv\Scripts\python.exe -m src.data_pipeline
venv\Scripts\python.exe -m src.feature_engineering
venv\Scripts\python.exe -m src.models.train
venv\Scripts\python.exe -m src.fairness_audit
venv\Scripts\python.exe -m pytest tests -q
```

If you want, I can turn this into a `RUNBOOK.md` or a `scripts/run_full_pipeline.ps1` so you can execute the whole flow with one command.
