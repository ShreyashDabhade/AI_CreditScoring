from src.query_parser import IntentType, ParsedQuery, parse_query
from src.sql_generator import build_sql_prompt, fallback_sql, generate_sql


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModel:
    def generate_content(self, prompt: str):
        assert "Applicants table schema:" in prompt
        return _FakeResponse("```sql\nSELECT * FROM applicants LIMIT 50\n```")


class _BrokenModel:
    def generate_content(self, prompt: str):
        raise RuntimeError("Gemini unavailable")


def test_build_sql_prompt_contains_schema_rules_and_context():
    parsed = ParsedQuery(
        intent=IntentType.FILTER_APPLICANTS.value,
        applicant_ids=[],
        filters=["income > 50000"],
        columns_mentioned=["income"],
        raw_query="show applicants with income > 50000",
        is_general=False,
    )

    prompt = build_sql_prompt(parsed)

    assert "Applicants table schema:" in prompt
    assert "Table name is always: applicants" in prompt
    assert "Detected intent: FILTER_APPLICANTS" in prompt
    assert "income > 50000" in prompt
    assert "show applicants with income > 50000" in prompt
    assert "Return only the raw SQL query. No markdown. No explanation." in prompt


def test_build_sql_prompt_resolves_sk_id_aliases_for_gemini():
    parsed = ParsedQuery(
        intent=IntentType.LOOKUP_APPLICANT.value,
        applicant_ids=[99999],
        filters=[],
        columns_mentioned=["applicant_id"],
        raw_query="Get the applicant of SK_ID = 99999",
        is_general=False,
    )

    prompt = build_sql_prompt(parsed)

    assert "SK_ID_CURR = 99999" in prompt
    assert "Table name is always: applicants" in prompt
    assert "- For LOOKUP: SELECT * FROM applicants WHERE SK_ID_CURR = <id>" in prompt
    assert 'SK_ID_CURR AS "Applicant ID"' in prompt
    assert "11. If the user query contains a numeric ID" in prompt


def test_fallback_sql_supports_lookup_and_explain():
    lookup = parse_query("show me applicant 12345")
    explain = parse_query("why was applicant 99999 rejected?")
    aggregate = parse_query("average income of rejected applicants")
    top_reason = parse_query("what is the top rejection reason for applicants under 25?")

    assert fallback_sql(lookup) == "SELECT * FROM applicants WHERE SK_ID_CURR = 12345 LIMIT 1;"
    assert (
        fallback_sql(explain)
        == "SELECT SK_ID_CURR, prediction, credit_score, risk_band, probability, "
           "AMT_INCOME_TOTAL_CAPPED, AMT_CREDIT, AGE_YEARS, "
           "top_features, explanation_text FROM applicants WHERE SK_ID_CURR = 99999 LIMIT 1;"
    )
    assert (
        fallback_sql(aggregate)
        == "SELECT AVG(AMT_INCOME_TOTAL_CAPPED) as avg_income FROM applicants WHERE prediction = 'REJECT';"
    )
    assert (
        fallback_sql(top_reason)
        == "SELECT top_features, COUNT(*) as freq FROM applicants "
           "WHERE prediction = 'REJECT' AND AGE_YEARS < 25 "
           "AND top_features IS NOT NULL GROUP BY top_features ORDER BY freq DESC LIMIT 10;"
    )


def test_fallback_sql_supports_simple_filter_queries():
    parsed = parse_query("show applicants with salary greater than 50000")

    sql = fallback_sql(parsed)

    assert sql is not None
    assert 'SK_ID_CURR AS "Applicant ID"' in sql
    assert 'AGE_YEARS AS "Age"' in sql
    assert 'credit_score AS "Internal Credit Score"' in sql
    assert "AMT_INCOME_TOTAL_CAPPED > 50000" in sql
    assert "LIMIT 50;" in sql


def test_fallback_sql_supports_compare_queries():
    parsed = parse_query("compare applicant 100002 and 100038")

    sql = fallback_sql(parsed)

    assert sql == "SELECT * FROM applicants WHERE SK_ID_CURR IN (100002, 100038) LIMIT 50;"


def test_generate_sql_strips_markdown_fences_from_gemini_response():
    parsed = parse_query("compare applicant 100 and 200")

    sql = generate_sql(parsed, _FakeModel())

    assert sql == "SELECT * FROM applicants LIMIT 50"


def test_generate_sql_falls_back_gracefully_on_gemini_exception():
    parsed = parse_query("show me applicant 12345")

    sql = generate_sql(parsed, _BrokenModel())

    assert sql == "SELECT * FROM applicants WHERE SK_ID_CURR = 12345 LIMIT 1;"
