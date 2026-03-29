**Backend Freeze Runbook**

Run commands from the repo root.

**Frozen Defaults**

- Runtime processed path: `data/processed/`
- Runtime artifact path: `artifacts/`
- Supported split modes: `proxy_time`, `random_stratified`
- Default split mode: `proxy_time`

**1. Install**

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

**2. Rebuild The Canonical Runtime Stack**

```bash
python -m src.data_pipeline
python -m src.feature_engineering
python -m src.models.train
python -m src.fairness_audit
```

Notes:

- Module 5 must continue to read only the canonical runtime names under `artifacts/`.
- Do not swap in files from offline experiment folders.
- Offline experiment folders are preserved for analysis, not deployment.

**3. Start The Real API**

```bash
python -c "from src.api.app import create_app; app = create_app(mock_mode=False, strict_artifacts=True); app.run(host='127.0.0.1', port=5000)"
```

Useful checks:

```bash
curl http://127.0.0.1:5000/health
```

Open the demo at:

- `http://127.0.0.1:5000/demo`

**4. Focused Merge-Readiness Checks**

```bash
python -m pytest tests/test_api.py
python -m pytest tests/test_backend_freeze_contract.py
python -m pytest tests/test_module1_split_regime.py
```

**5. Offline-Only Evidence**

Preserved offline paths include:

- `artifacts/reduced_thin_blend/`
- `artifacts/reduced_thin_lgbm/`
- `artifacts/reduced_thin_diag/`
- `artifacts/reduced_thin_opt/`
- `artifacts/alt_stacked_reduced/`
- `artifacts/random_stratified_reduced/`
- `artifacts/random_stratified_full_and_reduced/`

Those paths are not runtime API inputs.

**6. Docs To Hand To Frontend**

- `Documentation/backend_api_contract.md`
- `Documentation/backend_freeze_summary.md`
- `Documentation/artifact_contract.md`
- `Documentation/offline_experiment_index.md`
