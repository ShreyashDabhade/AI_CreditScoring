# Recovery Notes

## 2026-03-18 23:58:39

- Quarantined shared builder artifacts due confirmed builder artifact poisoning.
- Quarantine directory: `artifacts/quarantine/20260318_235839/`
- Moved files:
  - `artifacts/full_feature_builder.joblib`
  - `artifacts/reduced_feature_builder.joblib`
- Rule applied: quarantined artifacts are evidence only and must not be loaded or overwritten by normal repo flows.
