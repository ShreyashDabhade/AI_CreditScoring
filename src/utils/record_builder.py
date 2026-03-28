import pandas as pd
import json


def build_applicant_record(applicant_id, raw_features, probability, prediction, shap_values, feature_names):
    """Build a flat dict for SQLite insertion from model outputs.

    Returns a dict with exact keys:
      applicant_id, income, age, [all feature_names], probability,
      prediction, credit_score, risk_band, top_features, explanation_text

    Notes:
    - Does not mutate `raw_features`.
    - Uses only pandas and json.
    - Rounds floats to 4 decimals.
    """

    # Copy/normalize inputs
    feature_names = list(feature_names)

    if isinstance(raw_features, pd.Series):
        features = raw_features.to_dict()
    elif isinstance(raw_features, dict):
        features = dict(raw_features)
    else:
        # handle DataFrame row or other mapping-like
        features = dict(pd.Series(raw_features))

    out = {}
    out["applicant_id"] = applicant_id

    def _round_or_none(v):
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        try:
            fv = float(v)
        except Exception:
            return v
        return float(round(fv, 4))

    # income and age fields (may be present in features)
    out["income"] = _round_or_none(features.get("income"))
    out["age"] = _round_or_none(features.get("age"))

    # include all feature columns (preserve given names)
    for fname in feature_names:
        out[fname] = _round_or_none(features.get(fname))

    # probability (rounded to 4 decimals)
    prob = None if probability is None else float(probability)
    out["probability"] = None if prob is None else float(round(prob, 4))

    # prediction as int
    out["prediction"] = None if prediction is None else int(prediction)

    # credit_score: round((1 - probability) * 850), clamp [300,850]
    if prob is None:
        out["credit_score"] = None
    else:
        cs = round((1.0 - prob) * 850.0)
        cs = int(max(300, min(850, cs)))
        out["credit_score"] = cs

    # risk band
    if prob is None:
        out["risk_band"] = None
    else:
        if prob < 0.3:
            out["risk_band"] = "LOW"
        elif prob < 0.6:
            out["risk_band"] = "MEDIUM"
        else:
            out["risk_band"] = "HIGH"

    # prepare shap values: align length to feature_names, treat NaN/None as 0.0
    shap_list = list(shap_values) if shap_values is not None else []
    n = len(feature_names)
    aligned_shap = []
    for i in range(n):
        v = shap_list[i] if i < len(shap_list) else 0.0
        try:
            if pd.isna(v):
                v = 0.0
        except Exception:
            pass
        try:
            v = float(v)
        except Exception:
            v = 0.0
        aligned_shap.append(v)

    # top 5 features by absolute shap value
    indices = sorted(range(n), key=lambda i: abs(aligned_shap[i]), reverse=True)
    top = []
    for idx in indices[:5]:
        name = feature_names[idx]
        sv = float(round(aligned_shap[idx], 4))
        direction = "increases_risk" if sv > 0 else "decreases_risk"
        top.append({"feature": name, "shap_value": sv, "direction": direction})

    out["top_features"] = json.dumps(top)

    # explanation_text: 1-2 sentence readable summary using top 3 drivers
    top3 = top[:3]
    if not top3:
        explanation = "No explanation available."
    else:
        snippets = []
        for t in top3:
            desc = "increases risk" if t["direction"] == "increases_risk" else "decreases risk"
            snippets.append(f"{t['feature']} ({desc})")
        band = out.get("risk_band") or "unknown"
        cs_text = out.get("credit_score") if out.get("credit_score") is not None else "unknown"
        explanation = f"This applicant is in the {band} risk band (credit score {cs_text}). Top contributors: {', '.join(snippets)}."

    out["explanation_text"] = explanation

    return out
