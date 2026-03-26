## Module 1 — Data Pipeline: Integration Notes

### What this module does
Loads 7 raw Home Credit CSVs, enforces an allowlist, applies 6
data traps (sentinel removal, schema enforcement, income capping),
sorts by proxy staleness (oldest to newest), splits 60/10/10/20, and serializes 5
DataFrames plus a scalar income_cap to disk.

### What it expects from other modules
Nothing. Module 1 has zero upstream dependencies.

### What other modules can import from it
from src.data_pipeline import (
  enforce_locked_tables, enforce_schema, proxy_recency_sort,
  ordered_split_60_10_10_20, fit_missing_policy,
  apply_missing_policy, build_adversarial_dataset,
  serialize_dataframe, TRAIN_SCHEMA, LOCKED_SCORING_TABLES,
  ADVERSARIAL_ONLY_TABLES, PREV_SENTINEL_DAY_COLS,
  RELATIVE_TIME_COLS
)

### Constraints the integration AI must be aware of
1. income_cap MUST be loaded from data/processed/income_cap.joblib
   before calling any feature builder — never recompute it from
   non-train data.
2. app_test_adv.pkl is a dict, not a DataFrame. Load with:
   d = pickle.load(open(path,"rb"))
   adv_train, adv_val = d["adv_train"], d["adv_val"]
3. Trap C (AMT_INCOME_TOTAL_CAPPED) is already present in all
   4 split DataFrames — do not reapply.
4. DAYS_EMPLOYED_ANOM and all *_ANOM columns are already in
   train.pkl — do not recreate them in Module 2.
5. application_test.csv rows are in adv_train/adv_val only.
   They must never enter any scoring feature matrix.

