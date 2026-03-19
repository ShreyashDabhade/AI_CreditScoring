## Module 1 - Data Pipeline: Integration Notes

### What this module does
Loads the Home Credit application tables, applies schema enforcement and sentinel handling, sorts the cleaned application train data by proxy staleness, splits it `60/10/10/20` as `train -> val_model -> val_policy -> test = oldest -> newest`, and serializes the processed artifacts plus provenance metadata.

### What it expects from other modules
Nothing. Module 1 has zero upstream dependencies.

### What other modules can import from it
```python
from src.data_pipeline import (
  enforce_locked_tables,
  enforce_schema,
  proxy_recency_sort,
  ordered_split_60_10_10_20,
  fit_missing_policy,
  apply_missing_policy,
  build_adversarial_dataset,
  TRAIN_SCHEMA,
  LOCKED_SCORING_TABLES,
  ADVERSARIAL_ONLY_TABLES,
  PREV_SENTINEL_DAY_COLS,
  RELATIVE_TIME_COLS,
)
```

### Contracts downstream modules must respect
1. `income_cap.joblib` is authoritative and must be reused. Never recompute the cap from non-train data.
2. `processed_artifact_manifest.json` is authoritative provenance. Training and builder loading must validate against it.
3. `train.pkl`, `val_model.pkl`, `val_policy.pkl`, and `test.pkl` are the only scoring splits. They share identical schema, column order, and dtypes.
4. `AMT_INCOME_TOTAL_CAPPED` and `DAYS_EMPLOYED_ANOM` are already present in the scoring splits. Do not recreate them downstream.
5. `app_test_adv.pkl` is diagnostic-only. Load with:
   `d = pickle.load(open(path, "rb")); adv_train = d["adv_train"]; adv_val = d["adv_val"]`
6. Each adversarial half excludes `TARGET`, includes `ADV_LABEL`, `AMT_INCOME_TOTAL_CAPPED`, and `DAYS_EMPLOYED_ANOM`, and matches the non-label application column order.
7. `application_test.csv` rows must never enter any scoring feature matrix.

### Output set
- `data/processed/train.pkl`
- `data/processed/val_model.pkl`
- `data/processed/val_policy.pkl`
- `data/processed/test.pkl`
- `data/processed/app_test_adv.pkl`
- `data/processed/income_cap.joblib`
- `data/processed/processed_artifact_manifest.json`
- `data/data_quality_report.json`
