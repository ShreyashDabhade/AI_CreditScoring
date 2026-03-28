from src.query_parser import (
    IntentType,
    ParsedQuery,
    extract_applicant_ids,
    extract_filters,
    parse_natural_filter,
    parse_intent,
    parse_query,
)


def test_extract_applicant_ids_is_context_aware():
    assert extract_applicant_ids("Show applicants with income > 50000 and age < 30") == []
    assert extract_applicant_ids("Show me applicant 12345") == [12345]
    assert extract_applicant_ids("Compare applicant 100 and 200") == [100, 200]
    assert extract_applicant_ids("get applicant SK_ID= 99999") == [99999]


def test_extract_filters_resolves_aliases_to_real_columns():
    filters = extract_filters("income > 50000 and age < 30 and risk = high and score between 400 and 600")

    assert "income > 50000" in filters
    assert "age < 30" in filters
    assert "risk_band = HIGH" in filters
    assert "credit_score BETWEEN 400 AND 600" in filters


def test_parse_natural_filter_normalizes_plain_english_comparisons():
    assert parse_natural_filter("salary greater than 50000") == "AMT_INCOME_TOTAL_CAPPED > 50000"
    assert parse_natural_filter("applicants with income more than 80000") == "applicants with AMT_INCOME_TOTAL_CAPPED > 80000"
    assert parse_natural_filter("age less than 30") == "AGE_YEARS < 30"
    assert parse_natural_filter("score at least 600") == "credit_score >= 600"
    assert parse_natural_filter("AMT_INCOME_TOTAL_CAPPED > 50000") == "AMT_INCOME_TOTAL_CAPPED > 50000"


def test_parse_query_returns_lookup_dataclass():
    parsed = parse_query("show me applicant 12345")

    assert isinstance(parsed, ParsedQuery)
    assert parsed.intent == IntentType.LOOKUP_APPLICANT.value
    assert parsed.applicant_ids == [12345]
    assert "applicant_id" in parsed.columns_mentioned
    assert parsed.is_general is False


def test_parse_query_detects_new_letter_and_improvement_intents():
    letter = parse_query("generate a decision letter for applicant 100038")
    improvement = parse_query("what would applicant 100038 need to change to qualify?")

    assert letter.intent == IntentType.DECISION_LETTER.value
    assert letter.applicant_ids == [100038]
    assert improvement.intent == IntentType.IMPROVEMENT_SUGGESTION.value
    assert improvement.applicant_ids == [100038]


def test_parse_query_detects_aggregate_queries_with_filters():
    parsed = parse_query("average income of rejected applicants")

    assert parsed.intent == IntentType.AGGREGATE_QUERY.value
    assert "AMT_INCOME_TOTAL_CAPPED" in parsed.columns_mentioned
    assert "prediction = 1" in parsed.filters


def test_parse_query_detects_how_many_as_aggregate():
    parsed = parse_query("how many applicants were rejected?")

    assert parsed.intent == IntentType.AGGREGATE_QUERY.value


def test_parse_query_detects_general_questions():
    parsed = parse_query("What is a credit score?")

    assert parsed.intent == IntentType.GENERAL_QUESTION.value
    assert parsed.is_general is True
    assert "credit_score" in parsed.columns_mentioned


def test_parse_intent_keeps_legacy_shape():
    comparison = parse_intent("Compare applicant 100 and 200")
    explanation = parse_intent("Why was applicant 99999 rejected?")
    filtered = parse_intent("income greater than 1,000,000 and age under 25")

    assert comparison["intent"] == "COMPARISON_QUERY"
    assert comparison["applicant_ids"] == [100, 200]
    assert comparison["is_comparison"] is True

    assert explanation["intent"] == "EXPLANATION_QUERY"
    assert explanation["needs_explanation"] is True
    assert explanation["applicant_ids"] == [99999]

    assert filtered["intent"] == "FILTER_QUERY"
    assert filtered["filters"]["income"][">"] == 1000000
    assert filtered["filters"]["age"]["<"] == 25
