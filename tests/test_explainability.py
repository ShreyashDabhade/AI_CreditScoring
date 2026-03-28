from __future__ import annotations

from src.explainability import (
    adverse_action_reason_for_feature,
    build_adverse_action_report,
)


def test_adverse_action_reason_for_feature_maps_known_prefixes():
    bureau_reason = adverse_action_reason_for_feature("BUREAU_LOAN_COUNT")
    assert bureau_reason["title"] == "External credit obligations"
    assert bureau_reason["category"] == "bureau_history"

    ext_reason = adverse_action_reason_for_feature("EXT_SOURCE_2")
    assert ext_reason["title"] == "External credit reference strength"
    assert "external credit reference score" in ext_reason["detail"].lower()


def test_build_adverse_action_report_switches_tone_by_decision():
    explanations = [
        {"feature": "EXT_SOURCE_2", "reason": "External reference signals were weaker than preferred"},
        {"feature": "AMT_CREDIT", "reason": "Requested credit is high relative to stated income"},
        {"feature": "INST_DPD_MEAN", "reason": "Past installment payments show late-payment behavior"},
    ]

    adverse = build_adverse_action_report(explanations, "DECLINE")
    assert adverse["section_title"] == "Adverse Action Summary"
    assert adverse["emphasize_adverse"] is True
    assert len(adverse["reasons"]) == 3

    watch = build_adverse_action_report(explanations, "APPROVE")
    assert watch["section_title"] == "Primary Watch-Outs"
    assert watch["emphasize_adverse"] is False
