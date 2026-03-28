"""Demo client that inserts a sample applicant and calls the agent endpoints.

Usage:
    python scripts/demo_agent_request.py
"""
import json
import os
import sys
# Ensure repository root is on sys.path so `src` can be imported when running as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.api.app import create_app


def make_sample_record(applicant_id: int = 12345) -> dict:
    # Minimal record including fixed columns and one example dynamic feature
    top_features = [
        {"feature": "EXT_SOURCE_1", "shap_value": 0.42, "direction": "decreases_risk"},
        {"feature": "CREDIT_INCOME_RATIO", "shap_value": -0.31, "direction": "increases_risk"},
    ]
    return {
        "applicant_id": applicant_id,
        "income": 750000.0,
        "age": 34.0,
        "probability": 0.1234,
        "prediction": 0,
        "credit_score": 720,
        "risk_band": "LOW",
        "top_features": json.dumps(top_features),
        "explanation_text": "Top drivers identified",
        # dynamic feature placeholder (may be replaced by caller)
        # "dynamic_feature": 0.5,
    }


def pretty(v):
    try:
        return json.dumps(v, indent=2)
    except Exception:
        return str(v)


def main():
    app = create_app(mock_mode=True)
    client = app.test_client()

    # Insert a sample applicant into DB
    from src import db_loader

    rec = make_sample_record(99999)
    # choose an existing dynamic column from app config to avoid schema mismatch
    column_names = app.config.get("COLUMN_NAMES", [])
    fixed_count = 9  # number of FIXED_COLUMNS in db_schema.FIXED_COLUMNS
    dynamic_key = None
    if len(column_names) > fixed_count:
        dynamic_key = column_names[fixed_count]
    if dynamic_key:
        rec[dynamic_key] = 0.5
    db_path = app.config.get("DB_PATH")
    ok = db_loader.insert_applicant(rec, db_path)
    print("Inserted sample applicant?:", ok)

    # 1) Query endpoint
    qresp = client.post("/agent/query", json={"query": "Get applicant 99999"})
    print("/agent/query -> status", qresp.status_code)
    print(pretty(qresp.get_json()))

    # 2) Get applicant endpoint
    gres = client.get(f"/agent/applicant/99999")
    print("/agent/applicant/99999 -> status", gres.status_code)
    print(pretty(gres.get_json()))

    # 3) Explain endpoint
    eres = client.post("/agent/explain", json={"applicant_id": 99999})
    print("/agent/explain -> status", eres.status_code)
    print(pretty(eres.get_json()))


if __name__ == "__main__":
    main()
