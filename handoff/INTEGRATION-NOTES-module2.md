## Module 2 - Feature Engineering: Integration Notes

### What this module does
Splits feature engineering into a train-only `fit` stage and a frozen `transform` stage.

- `fit_full_builder(train_df, raw_dir, artifact_path=None)` builds the FULL pre-model train frame, fits rare-category policy, fits missing-flag eligibility at `>=5%`, fits numeric imputers, freezes encoded output columns, and only persists a builder artifact when an explicit `artifact_path` is provided.
- `fit_reduced_builder(train_df, artifact_path=None)` does the same for REDUCED and only persists when an explicit artifact path is provided.
- `build_full(df, builder, raw_dir)` and `build_reduced(df, builder)` are transform-only wrappers. They never refit policy.

### What this module expects from other modules
- From Module 1:
  - processed train data must exist for final Stage 4 acceptance: `data/processed/train.pkl`
  - full offline validation also expects `val_model.pkl`, `val_policy.pkl`, and `test.pkl`
  - application rows passed into Module 2 must already contain `AMT_INCOME_TOTAL_CAPPED` and `DAYS_EMPLOYED_ANOM`
- From configs:
  - rare-category threshold is locked to `500`
  - missing-flag eligibility is locked to `>= 0.05` missing rate on the final train pre-model frame

### What other modules should call
```python
from src.feature_engineering import (
    fit_full_builder,
    fit_reduced_builder,
    build_full,
    build_reduced,
)

reduced_builder = fit_reduced_builder(train_df)
full_builder = fit_full_builder(train_df, raw_dir="data/raw/")

X_train_reduced = build_reduced(train_df, reduced_builder)
X_train_full = build_full(train_df, full_builder, raw_dir="data/raw/")
```

### FULL runtime path
- FULL runtime supports flattened aggregate inputs.
- If all 45 aggregate feature columns are already present in the DataFrame, `build_full(..., raw_dir=None)` uses them directly and does not touch raw child CSVs.
- If flattened aggregate columns are absent, FULL offline transforms require `raw_dir` and `SK_ID_CURR` so Module 2 can merge aggregate features from the raw tables.
- Partial flattened aggregate input is rejected.

### Important behavior guarantees
- Transform never refits rare maps, missing-flag eligibility, imputers, encoded columns, or scaler.
- Output columns are aligned exactly to `builder.encoded_columns_`.
- Fairness raw columns and fairness-derived missing flags never appear in the model matrix.
- Scaling is applied only to frozen numeric feature columns, never to one-hot dummy columns or `_IS_MISSING` flags.
- Persisted builders are now paired with manifest files and must be loaded through the shared builder-artifact validator.

### Known blockers and limitations
1. Final Stage 4 acceptance is blocked until Module 1 provides `data/processed/train.pkl`.
2. Full offline Stage 4 validation is blocked until Module 1 also provides `val_model.pkl`, `val_policy.pkl`, and `test.pkl`.
3. FULL fit currently uses raw child CSV merges during training; FULL runtime inference should prefer flattened aggregate inputs so `/score` does not depend on raw CSV access.
