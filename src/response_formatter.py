import json
from typing import Any, Dict, List, Optional


def _first_present(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def format_applicant_row(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Applicant id - accept multiple possible keys
        applicant_id = _first_present(row, [
            "applicant_id",
            "applicant",
            "SK_ID_CURR",
            "sk_id_curr",
            "id",
        ])
        try:
            applicant_id = int(applicant_id) if applicant_id is not None else None
        except Exception:
            applicant_id = None

        # Income
        income = _first_present(row, ["income", "amt_income_total", "AMT_INCOME_TOTAL"])
        try:
            income = float(income) if income is not None else None
        except Exception:
            income = None

        # Age
        age = _first_present(row, ["age", "AGE"])
        try:
            age = float(age) if age is not None else None
        except Exception:
            age = None

        # Credit score
        credit_score = _first_present(row, ["credit_score", "creditScore", "score"])
        try:
            credit_score = int(credit_score) if credit_score is not None else None
        except Exception:
            credit_score = None

        # Risk band
        risk_band = _first_present(row, ["risk_band", "risk", "riskBand"]) or None
        if risk_band is not None:
            risk_band = str(risk_band)

        # Prediction mapping
        pred_raw = _first_present(row, ["prediction", "pred", "y_pred", "default"])
        prediction = None
        try:
            if pred_raw is None:
                prediction = None
            elif isinstance(pred_raw, (int, float)):
                prediction = "DEFAULT" if int(pred_raw) == 1 else "NO DEFAULT"
            else:
                s = str(pred_raw).strip().lower()
                if s in ("1", "true", "default", "yes", "y"):
                    prediction = "DEFAULT"
                elif s in ("0", "false", "no", "no default", "none", "n"):
                    prediction = "NO DEFAULT"
                else:
                    prediction = None
        except Exception:
            prediction = None

        # Probability (round to 4 decimals)
        prob_raw = _first_present(row, ["probability", "prob", "score", "probability_score"])
        probability = None
        try:
            if prob_raw is not None:
                probability = float(prob_raw)
                probability = round(probability, 4)
        except Exception:
            probability = None

        # Top features: may be JSON string or list
        top_raw = _first_present(row, ["top_features", "top_features_json", "top_features_str"])
        top_features: List[Dict[str, Any]] = []
        try:
            if top_raw is None:
                top_features = []
            elif isinstance(top_raw, str):
                try:
                    parsed = json.loads(top_raw)
                    if isinstance(parsed, list):
                        top_features = parsed
                    elif isinstance(parsed, dict):
                        top_features = [parsed]
                    else:
                        top_features = []
                except Exception:
                    top_features = []
            elif isinstance(top_raw, list):
                top_features = top_raw
            elif isinstance(top_raw, dict):
                top_features = [top_raw]
            else:
                top_features = []
        except Exception:
            top_features = []

        # Explanation text
        explanation_text = _first_present(row, ["explanation_text", "explanation", "explain_text"]) or None
        if explanation_text is not None:
            explanation_text = str(explanation_text)

        return {
            "applicant_id": applicant_id,
            "income": income,
            "age": age,
            "credit_score": credit_score,
            "risk_band": risk_band,
            "prediction": prediction,
            "probability": probability,
            "top_features": top_features,
            "explanation_text": explanation_text,
        }
    except Exception:
        # Never raise; return minimal sanitized structure
        return {
            "applicant_id": None,
            "income": None,
            "age": None,
            "credit_score": None,
            "risk_band": None,
            "prediction": None,
            "probability": None,
            "top_features": [],
            "explanation_text": None,
        }


def format_error(error_code: str, detail: str = "") -> Dict[str, Any]:
    mapping = {
        "INVALID_QUERY": "Your query could not be understood. Please rephrase.",
        "NO_RESULTS": "No matching applicants found.",
        "UNSAFE_SQL": "This query is not permitted.",
        "AGENT_ERROR": "An internal error occurred. Please try again.",
    }
    message = mapping.get(error_code, "An error occurred.")
    if detail:
        # keep messages concise; append short detail
        try:
            detail_str = str(detail)
            # Only append small bits of detail
            if len(detail_str) > 0:
                message = f"{message} ({detail_str})"
        except Exception:
            pass

    return {"status": "error", "error_code": error_code, "message": message}


def _normalize_rows(raw: Any) -> List[Dict[str, Any]]:
    # Attempt to convert various possible raw result formats into a list of plain dicts
    try:
        if raw is None:
            return []
        # If it's already a list of dicts
        if isinstance(raw, list):
            out: List[Dict[str, Any]] = []
            for item in raw:
                if isinstance(item, dict):
                    out.append(item)
                elif hasattr(item, "to_dict") and callable(item.to_dict):
                    try:
                        out.append(item.to_dict())
                    except Exception:
                        # fallback to str representation
                        out.append({"value": str(item)})
                else:
                    out.append({"value": item})
            return out

        # If it's an object like a DataFrame with to_dict(orient='records')
        if hasattr(raw, "to_dict") and callable(raw.to_dict):
            try:
                records = raw.to_dict(orient="records")
            except Exception:
                try:
                    records = raw.to_dict()
                except Exception:
                    records = None
            if isinstance(records, list):
                return [r if isinstance(r, dict) else {"value": r} for r in records]
            # If dict-of-lists, convert to records
            if isinstance(records, dict):
                # columns -> lists
                keys = list(records.keys())
                length = 0
                for k in keys:
                    v = records.get(k)
                    if hasattr(v, "__len__"):
                        length = max(length, len(v))
                out = []
                for i in range(length):
                    row = {}
                    for k in keys:
                        col = records.get(k)
                        try:
                            row[k] = col[i]
                        except Exception:
                            row[k] = None
                    out.append(row)
                return out

        # If it's an iterator/generator
        if hasattr(raw, "__iter__"):
            return _normalize_rows(list(raw))

        # Fallback: single value
        return [{"value": raw}]
    except Exception:
        return []


def format_response(agent_output: Dict[str, Any], intent: str, include_explanation: bool) -> Dict[str, Any]:
    try:
        if not isinstance(agent_output, dict):
            return format_error("AGENT_ERROR", "malformed agent output")

        # If agent reported an error, map it
        err_code = agent_output.get("error_code") or agent_output.get("error")
        if err_code:
            return format_error(str(err_code), agent_output.get("error_message") or agent_output.get("detail") or "")

        # Extract SQL query used
        query = agent_output.get("query") or agent_output.get("sql") or agent_output.get("generated_sql") or ""

        # Extract raw rows from common keys
        raw_candidates = [
            agent_output.get(k)
            for k in ("results", "rows", "data", "applicants", "rows_list", "df")
            if agent_output.get(k) is not None
        ]
        raw = raw_candidates[0] if raw_candidates else agent_output.get("results")

        rows = _normalize_rows(raw)

        applicants: List[Dict[str, Any]] = [format_applicant_row(r) for r in rows]

        # If no applicants found, return NO_RESULTS error per spec
        if not applicants:
            return format_error("NO_RESULTS")

        explanation = None
        if include_explanation:
            explanation = agent_output.get("explanation") or agent_output.get("shap_explanation") or None

        response = {
            "status": "success",
            "query": query if query is not None else "",
            "count": len(applicants),
            "intent": intent,
            "applicants": applicants,
            "explanation": explanation if explanation is not None else None,
        }

        return response
    except Exception as e:
        return format_error("AGENT_ERROR", str(e))
