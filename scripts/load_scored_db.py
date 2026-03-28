import os
import sqlite3
from pathlib import Path

import pandas as pd

# Exact scored export found in the repo.
SCORED_FILE = "logs/2026-03-28T12-57-06Z_full.csv"
DB_PATH = "data/applicants.db"


def main() -> None:
    if SCORED_FILE.endswith(".pkl"):
        df = pd.read_pickle(SCORED_FILE)
    else:
        df = pd.read_csv(SCORED_FILE)

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    required = ["SK_ID_CURR"]
    missing = [column_name for column_name in required if column_name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rename_map: dict[str, str] = {}
    if "prediction" not in df.columns and "decision" in df.columns:
        rename_map["decision"] = "prediction"
    if "probability" not in df.columns and "probability_default" in df.columns:
        rename_map["probability_default"] = "probability"
    if "prediction" not in df.columns and "TARGET" in df.columns:
        df["prediction"] = df["TARGET"].map({0: "APPROVE", 1: "REJECT"})
    if "probability" not in df.columns and "pred_proba" in df.columns:
        rename_map["pred_proba"] = "probability"
    if "probability" not in df.columns and "score" in df.columns:
        rename_map["score"] = "probability"
    if rename_map:
        df = df.rename(columns=rename_map)

    if df.columns.duplicated().any():
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        print(f"Dropping duplicate columns: {duplicate_columns}")
        df = df.loc[:, ~df.columns.duplicated()].copy()

    seen_columns: dict[str, int] = {}
    normalized_columns: list[str] = []
    renamed_columns: list[str] = []
    for column_name in df.columns:
        candidate = str(column_name)
        lowered = candidate.lower()
        if lowered in seen_columns:
            seen_columns[lowered] += 1
            candidate = f"{candidate}_dup{seen_columns[lowered]}"
            renamed_columns.append(candidate)
        else:
            seen_columns[lowered] = 0
        normalized_columns.append(candidate)
    if renamed_columns:
        print(f"Renamed case-colliding columns: {renamed_columns}")
        df.columns = normalized_columns

    if "probability" in df.columns:
        df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    if "credit_score" not in df.columns and "probability" in df.columns:
        df["credit_score"] = (850 - (df["probability"] * 550)).astype(int).clip(300, 850)

    if "risk_band" not in df.columns and "probability" in df.columns:
        def get_band(probability: float) -> str:
            if probability >= 0.6:
                return "HIGH"
            if probability >= 0.3:
                return "MEDIUM"
            return "LOW"

        df["risk_band"] = df["probability"].apply(get_band)

    if "prediction" not in df.columns and "probability" in df.columns:
        df["prediction"] = df["probability"].apply(
            lambda probability: "REJECT" if probability >= 0.5 else "APPROVE"
        )

    if "explanation_text" not in df.columns:
        df["explanation_text"] = df.apply(
            lambda row: (
                f"Applicant {row['SK_ID_CURR']} was "
                f"{'rejected' if row.get('prediction') == 'REJECT' else 'approved'} "
                f"with a default probability of "
                f"{row.get('probability', 0) * 100:.1f}%."
            ),
            axis=1,
        )

    if "top_features" not in df.columns:
        df["top_features"] = "Not available"

    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    backup_path = DB_PATH + ".bak"
    if os.path.exists(DB_PATH):
        import shutil

        shutil.copy2(DB_PATH, backup_path)
        print(f"Backed up existing DB to {backup_path}")

    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql("applicants", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sk ON applicants(SK_ID_CURR)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred ON applicants(prediction)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_risk ON applicants(risk_band)")
        conn.commit()
    except Exception:
        conn.close()
        if os.path.exists(backup_path):
            import shutil

            shutil.copy2(backup_path, DB_PATH)
            print(f"Restored backup from {backup_path}")
        raise

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM applicants")
    total = cur.fetchone()[0]
    cur.execute("SELECT prediction, COUNT(*) FROM applicants GROUP BY prediction")
    breakdown = dict(cur.fetchall())
    cur.execute("SELECT SK_ID_CURR FROM applicants LIMIT 5")
    sample_ids = [row[0] for row in cur.fetchall()]
    conn.close()

    print(f"DB written: {total} rows")
    print(f"Prediction breakdown: {breakdown}")
    print(f"Sample IDs: {sample_ids}")
    print("Done.")


if __name__ == "__main__":
    main()
