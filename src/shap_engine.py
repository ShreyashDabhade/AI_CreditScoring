"""SHAP formatting and explanation utilities.

Functions:
- extract_top_features(applicant_record: dict) -> list[dict]
- generate_explanation(applicant_record: dict) -> dict
- compare_applicants(record_a: dict, record_b: dict) -> dict

This module parses stored `top_features` JSON (no SHAP computation here).
"""
from __future__ import annotations

import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def extract_top_features(applicant_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse `top_features` JSON string and return top 5 feature dicts with rank.

    Each dict: {"feature": str, "shap_value": float, "direction": str, "rank": int}
    Sort by abs(shap_value) descending. Handle malformed or missing JSON gracefully.
    """
    raw = applicant_record.get("top_features")
    if not raw:
        logger.warning("No top_features present for applicant %s", applicant_record.get("applicant_id"))
        return []

    parsed = None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            logger.warning("Malformed top_features JSON for applicant %s", applicant_record.get("applicant_id"))
            return []
    elif isinstance(raw, list):
        parsed = raw
    else:
        # unknown type
        logger.warning("Unsupported top_features type for applicant %s: %s", applicant_record.get("applicant_id"), type(raw))
        return []

    out = []
    for item in parsed:
        try:
            feat = item.get("feature") if isinstance(item, dict) else None
            sv = item.get("shap_value") if isinstance(item, dict) else None
            direction = item.get("direction") if isinstance(item, dict) else None
            sv = 0.0 if sv is None else float(sv)
            direction = str(direction) if direction is not None else ("increases_risk" if sv > 0 else "decreases_risk")
            out.append({"feature": feat, "shap_value": float(round(sv, 4)), "direction": direction})
        except Exception:
            # skip malformed entry
            continue

    # sort by absolute shap value descending and add rank
    out_sorted = sorted(out, key=lambda x: abs(x.get("shap_value", 0.0)), reverse=True)
    for idx, item in enumerate(out_sorted[:5], start=1):
        item["rank"] = idx

    return out_sorted[:5]


def generate_explanation(applicant_record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a structured explanation dict for a single applicant record."""
    aid = applicant_record.get("applicant_id")
    pred_raw = applicant_record.get("prediction")
    try:
        pred_flag = int(pred_raw) if pred_raw is not None else None
    except Exception:
        pred_flag = None

    prediction = "DEFAULT" if pred_flag == 1 else "NO DEFAULT"

    credit_score = applicant_record.get("credit_score")
    risk_band = applicant_record.get("risk_band")
    prob = applicant_record.get("probability")
    try:
        prob = None if prob is None else float(round(float(prob), 4))
    except Exception:
        prob = None

    top5 = extract_top_features(applicant_record)

    summary = applicant_record.get("explanation_text") or ""

    return {
        "applicant_id": aid,
        "prediction": prediction,
        "credit_score": credit_score,
        "risk_band": risk_band,
        "probability": prob,
        "top_5_reasons": top5,
        "summary": summary,
    }


def compare_applicants(record_a: Dict[str, Any], record_b: Dict[str, Any]) -> Dict[str, Any]:
    """Return side-by-side explanations and a simple comparison note."""
    a = generate_explanation(record_a)
    b = generate_explanation(record_b)

    note_parts = []
    # compare risk band
    rb_a = a.get("risk_band") or "unknown"
    rb_b = b.get("risk_band") or "unknown"
    if rb_a != rb_b:
        note_parts.append(f"Applicant {a.get('applicant_id')} is {rb_a} risk vs {b.get('applicant_id')} at {rb_b}.")

    # compare credit score
    cs_a = a.get("credit_score")
    cs_b = b.get("credit_score")
    try:
        if cs_a is not None and cs_b is not None:
            diff = int(cs_a) - int(cs_b)
            if diff > 0:
                note_parts.append(f"Applicant {a.get('applicant_id')} has higher credit score by {diff} points.")
            elif diff < 0:
                note_parts.append(f"Applicant {b.get('applicant_id')} has higher credit score by {abs(diff)} points.")
    except Exception:
        pass

    comparison_note = " ".join(note_parts) if note_parts else "Applicants have similar risk profiles."

    return {"applicant_a": a, "applicant_b": b, "comparison_note": comparison_note}
