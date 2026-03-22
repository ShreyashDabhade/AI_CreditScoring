#!/usr/bin/env python
"""Test if Module 4 dependencies (M3 artifacts and real test data) can load."""
import os
import pickle
from src.models.train import load_artifacts

print("=" * 60)
print("Module 4 Dependency Check")
print("=" * 60)

# Test 1: Check artifact paths
artifact_dir = "artifacts/"
print(f"\n1. Artifact directory exists: {os.path.exists(artifact_dir)}")
files = os.listdir(artifact_dir)
joblib_files = sorted([f for f in files if f.endswith('.joblib')])
print(f"   Files in artifacts/: {joblib_files}")

# Test 2: Try loading artifacts
print(f"\n2. Loading Module 3 artifacts...")
try:
    artifacts = load_artifacts(artifact_dir)
    print(f"   ✓ Artifacts loaded successfully")
    for key in ['full_model', 'full_calibrator', 'full_shap_explainer']:
        print(f"     - {key}: {type(artifacts[key]).__name__}")
except Exception as e:
    print(f"   ✗ Error loading artifacts: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try loading real test data
print(f"\n3. Loading real test data...")
try:
    with open("data/processed/test.pkl", "rb") as f:
        test_df = pickle.load(f)
    print(f"   ✓ test.pkl loaded: shape={test_df.shape}")
    print(f"     Columns: {list(test_df.columns[:10])}...")
except Exception as e:
    print(f"   ✗ Error loading test.pkl: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All dependencies ready for Module 4")
print("=" * 60)
