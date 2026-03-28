**Real Pipeline Runbook**

Run commands from the repo root.

**Final Branch Truth**

- REDUCED runtime baseline remains XGBoost.
- REDUCED LightGBM and REDUCED weighted blend remain offline-only candidates.
- The offline REDUCED experiments are preserved for inspection, not deployment.

**1. Setup**

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Set runtime paths for the shell session:

```bash
export DATA_PROCESSED_DIR=data/processed/
export ARTIFACT_DIR=artifacts/
```

**2. Rebuild The Canonical Runtime Stack**

Module 1:

```bash
python -m src.data_pipeline
```

Module 2:

```bash
python -m src.feature_engineering
```

Module 3:

```bash
python -m src.models.train
```

Module 4:

```bash
python -m src.fairness_audit
```

Important:

- Do not replace `artifacts/reduced_model.joblib` or `artifacts/reduced_calibrator.joblib` with offline experiment artifacts.
- Module 5 should continue to read only the canonical runtime artifact names.

**3. Start The Real API**

```bash
python -c "from src.api.app import create_app; app = create_app(artifact_dir='artifacts', processed_dir='data/processed', mock_mode=False, strict_artifacts=True); app.run(host='127.0.0.1', port=5000)"
```

Useful checks:

```bash
curl http://127.0.0.1:5000/health
```

Open the demo at:

- `http://127.0.0.1:5000/demo`

**4. Reproduce The Offline REDUCED Diagnostics**

REDUCED blend experiment:

```bash
python -c "from src.models.reduced_blend import run_reduced_blend_experiment; run_reduced_blend_experiment()"
```

REDUCED calibrated comparison:

```bash
python -c "from src.models.reduced_blend_calibrated_compare import run_reduced_blend_calibrated_compare; run_reduced_blend_calibrated_compare()"
```

REDUCED policy-threshold diagnostic:

```bash
python -c "from src.models.reduced_policy_threshold_diag import run_reduced_policy_threshold_diag; run_reduced_policy_threshold_diag()"
```

REDUCED previous-vs-merged retraining comparison:

```bash
python -c "from src.models.reduced_training_regime_compare import run_reduced_training_regime_comparison; run_reduced_training_regime_comparison()"
```

FULL subgroup calibration experiment:

```bash
python -c "from src.models.subgroup_calibration import run_subgroup_calibration_experiments; run_subgroup_calibration_experiments()"
```

FULL fairness-aware retraining experiment:

```bash
python -c "from src.models.fairness_aware_training import run_fairness_aware_modeling_experiments; run_fairness_aware_modeling_experiments()"
```

Those reports write only to:

- `artifacts/reduced_blend/`
- `artifacts/reduced_blend_calibrated_compare/`
- `artifacts/reduced_policy_threshold_diag/`
- `artifacts/reduced_training_regime_comparison_report.json`
- `artifacts/subgroup_calibration_experiment_report.json`
- `artifacts/fairness_aware_modeling_experiment_report.json`

They are preserved as offline evidence only.

**5. Focused Regression Checks**

```bash
python -m pytest tests/test_api.py
python -m pytest tests/test_m3_blending.py
python -m pytest tests/test_reduced_blend_calibrated_compare.py
python -m pytest tests/test_reduced_policy_threshold_diag.py
```

**6. Where To Read The Final Story**

- `README.md`
- `Documentation/final_branch_summary.md`
- `Documentation/offline_experiment_index.md`
- `Documentation/artifact_contract.md`
