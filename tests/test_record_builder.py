from src.utils.record_builder import build_applicant_record


def _make_sample():
    feature_names = [f"feat_{i}" for i in range(1, 129)]
    raw_features = {name: float(i) for i, name in enumerate(feature_names, start=1)}
    # include explicit income and age keys
    raw_features["income"] = 55000.12345
    raw_features["age"] = 42
    # shap values: alternating sign
    shap = [((-1) ** i) * (i * 0.01) for i in range(1, 129)]
    return feature_names, raw_features, shap


def test_builder_runs():
    feature_names, raw_features, shap = _make_sample()
    rec = build_applicant_record(
        applicant_id=12345,
        raw_features=raw_features,
        probability=0.42,
        prediction=1,
        shap_values=shap,
        feature_names=feature_names,
    )

    assert rec["applicant_id"] == 12345
    assert isinstance(rec["probability"], float)
    assert rec["prediction"] == 1
    assert "feat_1" in rec and "feat_128" in rec
    assert isinstance(rec["top_features"], str)
    assert "explanation_text" in rec


if __name__ == "__main__":
    test_builder_runs()
    print("sanity ok")
