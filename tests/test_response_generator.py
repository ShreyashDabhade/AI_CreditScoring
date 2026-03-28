from src.query_parser import IntentType, ParsedQuery
from src.response_generator import (
    build_compare_prompt,
    build_decision_letter_prompt,
    build_explain_prompt,
    build_improvement_prompt,
    build_response_prompt,
    generate_response,
    handle_general_question,
)
from src.sql_executor import ExecutionResult


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModel:
    def __init__(self, text: str):
        self.text = text
        self.last_prompt = None

    def generate_content(self, prompt: str):
        self.last_prompt = prompt
        return _FakeResponse(self.text)


class _BrokenModel:
    def generate_content(self, prompt: str):
        raise RuntimeError("boom")


def _sample_parsed(intent: str = IntentType.FILTER_APPLICANTS.value) -> ParsedQuery:
    return ParsedQuery(
        intent=intent,
        applicant_ids=[12345] if intent == IntentType.EXPLAIN_APPLICANT.value else [],
        filters=["risk_band = high"] if intent == IntentType.FILTER_APPLICANTS.value else [],
        columns_mentioned=["risk_band", "credit_score"],
        raw_query="show high risk applicants",
        is_general=intent == IntentType.GENERAL_QUESTION.value,
    )


def _sample_result(row_count: int = 1) -> ExecutionResult:
    rows = [
        {
            "SK_ID_CURR": 100001 + index,
            "credit_score": 0.7 - (index * 0.1),
            "risk_band": "high" if index % 2 else "low",
            "explanation_text": "Example explanation text",
            "top_features": [{"feature": "credit_score"}],
        }
        for index in range(row_count)
    ]
    return ExecutionResult(
        success=True,
        rows=rows,
        row_count=len(rows),
        columns=["SK_ID_CURR", "credit_score", "risk_band", "explanation_text", "top_features"],
        error_message=None,
        sql_executed="SELECT * FROM applicants;",
    )


def test_build_response_prompt_includes_context_and_full_rows_for_small_results():
    prompt = build_response_prompt(
        _sample_parsed(IntentType.EXPLAIN_APPLICANT.value),
        _sample_result(1),
        "Why was applicant 12345 rejected?",
    )

    assert "You are a credit risk analyst assistant" in prompt
    assert "Original question: Why was applicant 12345 rejected?" in prompt
    assert "Detected intent: EXPLAIN_APPLICANT" in prompt
    assert "Returned rows (1):" in prompt
    assert "top_features" in prompt


def test_build_response_prompt_includes_summary_for_many_rows():
    prompt = build_response_prompt(
        _sample_parsed(),
        _sample_result(6),
        "show high risk applicants",
    )

    assert "First 5 rows:" in prompt
    assert "Summary stats:" in prompt
    assert "- Count: 6" in prompt
    assert "credit_score: min=" in prompt


def test_build_explain_prompt_formats_probability_and_employment_years():
    prompt = build_explain_prompt(
        {
            "SK_ID_CURR": 123654,
            "prediction": "REJECT",
            "credit_score": 420,
            "risk_band": "HIGH",
            "probability": 0.73,
            "AMT_INCOME_TOTAL_CAPPED": 67500,
            "AMT_CREDIT": 450000,
            "AGE_YEARS": 34,
            "DAYS_EMPLOYED": -1460,
            "top_features": "high debt-to-income ratio, short employment history, previous late payments",
            "explanation_text": "Applicant shows elevated default risk due to income-to-loan ratio exceeding safe thresholds",
        }
    )

    assert "Applicant ID    : 123654" in prompt
    assert "Default Probability: 73.0%" in prompt
    assert "Employment      : 4 years" in prompt
    assert "Loan Requested  : 450000" in prompt


def test_build_decision_letter_prompt_includes_formal_opening():
    prompt = build_decision_letter_prompt(
        {
            "SK_ID_CURR": 123654,
            "prediction": "REJECT",
            "credit_score": 420,
            "risk_band": "HIGH",
            "probability": 0.73,
            "AMT_INCOME_TOTAL_CAPPED": 67500,
            "AMT_CREDIT": 450000,
            "top_features": "high debt-to-income ratio, short employment history",
            "explanation_text": "Elevated default risk.",
        }
    )

    assert 'Dear Applicant (Ref: 123654),' in prompt
    assert "Default Risk  : 73.0%" in prompt


def test_build_improvement_prompt_short_circuits_for_approved_applicant():
    prompt = build_improvement_prompt(
        {
            "SK_ID_CURR": 100002,
            "prediction": "APPROVE",
            "credit_score": 710,
            "probability": 0.12,
        }
    )

    assert "No improvement needed." in prompt


def test_build_compare_prompt_includes_both_applicants():
    prompt = build_compare_prompt(
        [
            {
                "SK_ID_CURR": 100002,
                "prediction": "APPROVE",
                "credit_score": 710,
                "risk_band": "LOW",
                "probability": 0.12,
                "AMT_INCOME_TOTAL_CAPPED": 120000,
                "AMT_CREDIT": 250000,
                "AGE_YEARS": 36,
                "top_features": "stable income",
            },
            {
                "SK_ID_CURR": 100038,
                "prediction": "REJECT",
                "credit_score": 420,
                "risk_band": "HIGH",
                "probability": 0.73,
                "AMT_INCOME_TOTAL_CAPPED": 67500,
                "AMT_CREDIT": 450000,
                "AGE_YEARS": 34,
                "top_features": "high debt-to-income ratio",
            },
        ]
    )

    assert "APPLICANT A:" in prompt
    assert "APPLICANT B:" in prompt
    assert "ID: 100002" in prompt
    assert "ID: 100038" in prompt


def test_generate_response_uses_model_and_handles_general_questions():
    model = _FakeModel("Concise answer")

    text = generate_response(
        _sample_parsed(IntentType.GENERAL_QUESTION.value),
        ExecutionResult(success=True, rows=[], row_count=0, columns=[], error_message=None, sql_executed=""),
        "What is a credit score?",
        model,
    )

    assert text == "Concise answer"
    assert "Answer this question about credit scoring concisely" in model.last_prompt


def test_generate_response_uses_explain_prompt_for_explain_intent():
    model = _FakeModel("Narrative explanation")
    parsed = _sample_parsed(IntentType.EXPLAIN_APPLICANT.value)
    result = ExecutionResult(
        success=True,
        rows=[
            {
                "SK_ID_CURR": 123654,
                "prediction": "REJECT",
                "credit_score": 420,
                "risk_band": "HIGH",
                "probability": 0.73,
                "AMT_INCOME_TOTAL_CAPPED": 67500,
                "AMT_CREDIT": 450000,
                "AGE_YEARS": 34,
                "DAYS_EMPLOYED": -1460,
                "top_features": "high debt-to-income ratio, short employment history, previous late payments",
                "explanation_text": "Applicant shows elevated default risk due to income-to-loan ratio exceeding safe thresholds",
            }
        ],
        row_count=1,
        columns=[
            "SK_ID_CURR",
            "prediction",
            "credit_score",
            "risk_band",
            "probability",
            "AMT_INCOME_TOTAL_CAPPED",
            "AMT_CREDIT",
            "AGE_YEARS",
            "DAYS_EMPLOYED",
            "top_features",
            "explanation_text",
        ],
        error_message=None,
        sql_executed="SELECT * FROM applicants WHERE SK_ID_CURR = 123654;",
    )

    text = generate_response(parsed, result, "Why didn't applicant 123654 get a loan?", model)

    assert text == "Narrative explanation"
    assert "Applicant ID    : 123654" in model.last_prompt
    assert "Default Probability: 73.0%" in model.last_prompt


def test_generate_response_uses_compare_prompt_for_compare_intent():
    model = _FakeModel("Comparison narrative")
    parsed = ParsedQuery(
        intent=IntentType.COMPARE_APPLICANTS.value,
        applicant_ids=[100002, 100038],
        filters=[],
        columns_mentioned=["credit_score", "risk_band"],
        raw_query="compare applicant 100002 and 100038",
        is_general=False,
    )
    result = ExecutionResult(
        success=True,
        rows=[
            {
                "SK_ID_CURR": 100002,
                "prediction": "APPROVE",
                "credit_score": 710,
                "risk_band": "LOW",
                "probability": 0.12,
                "AMT_INCOME_TOTAL_CAPPED": 120000,
                "AMT_CREDIT": 250000,
                "AGE_YEARS": 36,
                "top_features": "stable income",
            },
            {
                "SK_ID_CURR": 100038,
                "prediction": "REJECT",
                "credit_score": 420,
                "risk_band": "HIGH",
                "probability": 0.73,
                "AMT_INCOME_TOTAL_CAPPED": 67500,
                "AMT_CREDIT": 450000,
                "AGE_YEARS": 34,
                "top_features": "high debt-to-income ratio",
            },
        ],
        row_count=2,
        columns=["SK_ID_CURR", "prediction", "credit_score", "risk_band"],
        error_message=None,
        sql_executed="SELECT * FROM applicants WHERE SK_ID_CURR IN (100002, 100038);",
    )

    text = generate_response(parsed, result, parsed.raw_query, model)

    assert text == "Comparison narrative"
    assert "APPLICANT A:" in model.last_prompt
    assert "APPLICANT B:" in model.last_prompt


def test_generate_response_uses_decision_letter_prompt():
    model = _FakeModel("Dear Applicant (Ref: 100038),")
    parsed = ParsedQuery(
        intent=IntentType.DECISION_LETTER.value,
        applicant_ids=[100038],
        filters=[],
        columns_mentioned=["prediction", "credit_score"],
        raw_query="generate a decision letter for applicant 100038",
        is_general=False,
    )
    result = ExecutionResult(
        success=True,
        rows=[{
            "SK_ID_CURR": 100038,
            "prediction": "REJECT",
            "credit_score": 420,
            "risk_band": "HIGH",
            "probability": 0.73,
            "AMT_INCOME_TOTAL_CAPPED": 67500,
            "AMT_CREDIT": 450000,
            "top_features": "high debt-to-income ratio",
            "explanation_text": "Elevated default risk.",
        }],
        row_count=1,
        columns=["SK_ID_CURR", "prediction"],
        error_message=None,
        sql_executed="SELECT * FROM applicants WHERE SK_ID_CURR = 100038;",
    )

    text = generate_response(parsed, result, parsed.raw_query, model)

    assert text == "Dear Applicant (Ref: 100038),"
    assert "Write the letter now:" in model.last_prompt


def test_generate_response_uses_improvement_prompt():
    model = _FakeModel("Constructive advice")
    parsed = ParsedQuery(
        intent=IntentType.IMPROVEMENT_SUGGESTION.value,
        applicant_ids=[100038],
        filters=[],
        columns_mentioned=["prediction", "credit_score"],
        raw_query="what would applicant 100038 need to change to qualify?",
        is_general=False,
    )
    result = ExecutionResult(
        success=True,
        rows=[{
            "SK_ID_CURR": 100038,
            "prediction": "REJECT",
            "credit_score": 420,
            "risk_band": "HIGH",
            "probability": 0.73,
            "top_features": "high debt-to-income ratio",
        }],
        row_count=1,
        columns=["SK_ID_CURR", "prediction"],
        error_message=None,
        sql_executed="SELECT * FROM applicants WHERE SK_ID_CURR = 100038;",
    )

    text = generate_response(parsed, result, parsed.raw_query, model)

    assert text == "Constructive advice"
    assert "what they need to improve to qualify in the future" in model.last_prompt


def test_generate_response_falls_back_to_explain_narrative_on_model_failure():
    parsed = ParsedQuery(
        intent=IntentType.EXPLAIN_APPLICANT.value,
        applicant_ids=[100038],
        filters=[],
        columns_mentioned=["prediction", "credit_score"],
        raw_query="why was applicant 100038 rejected?",
        is_general=False,
    )
    result = ExecutionResult(
        success=True,
        rows=[{
            "SK_ID_CURR": 100038,
            "prediction": "REJECT",
            "risk_band": "HIGH",
            "probability": 0.73,
            "top_features": "high debt-to-income ratio, short employment history",
            "explanation_text": "The application showed elevated repayment risk.",
        }],
        row_count=1,
        columns=["SK_ID_CURR", "prediction"],
        error_message=None,
        sql_executed="SELECT * FROM applicants WHERE SK_ID_CURR = 100038;",
    )

    text = generate_response(parsed, result, parsed.raw_query, _BrokenModel())

    assert "Applicant 100038 was rejected" in text
    assert "73.0%" in text


def test_generate_response_falls_back_to_letter_on_model_failure():
    parsed = ParsedQuery(
        intent=IntentType.DECISION_LETTER.value,
        applicant_ids=[100038],
        filters=[],
        columns_mentioned=["prediction"],
        raw_query="generate a decision letter for applicant 100038",
        is_general=False,
    )
    result = ExecutionResult(
        success=True,
        rows=[{
            "SK_ID_CURR": 100038,
            "prediction": "REJECT",
            "risk_band": "HIGH",
            "probability": 0.73,
            "AMT_CREDIT": 450000,
            "top_features": "high debt-to-income ratio",
        }],
        row_count=1,
        columns=["SK_ID_CURR", "prediction"],
        error_message=None,
        sql_executed="SELECT * FROM applicants WHERE SK_ID_CURR = 100038;",
    )

    text = generate_response(parsed, result, parsed.raw_query, _BrokenModel())

    assert text.startswith("Dear Applicant (Ref: 100038),")
    assert "Credit Risk Department" in text


def test_generate_response_falls_back_to_improvement_guidance_on_model_failure():
    parsed = ParsedQuery(
        intent=IntentType.IMPROVEMENT_SUGGESTION.value,
        applicant_ids=[100038],
        filters=[],
        columns_mentioned=["prediction"],
        raw_query="what would applicant 100038 need to change to qualify?",
        is_general=False,
    )
    result = ExecutionResult(
        success=True,
        rows=[{
            "SK_ID_CURR": 100038,
            "prediction": "REJECT",
            "credit_score": 420,
            "top_features": "high debt-to-income ratio, previous late payments",
        }],
        row_count=1,
        columns=["SK_ID_CURR", "prediction"],
        error_message=None,
        sql_executed="SELECT * FROM applicants WHERE SK_ID_CURR = 100038;",
    )

    text = generate_response(parsed, result, parsed.raw_query, _BrokenModel())

    assert "Applicant 100038 was not approved this time" in text
    assert "reapply" in text


def test_generate_response_falls_back_to_compare_narrative_on_model_failure():
    parsed = ParsedQuery(
        intent=IntentType.COMPARE_APPLICANTS.value,
        applicant_ids=[100002, 100038],
        filters=[],
        columns_mentioned=["credit_score", "risk_band"],
        raw_query="compare applicant 100002 and 100038",
        is_general=False,
    )
    result = ExecutionResult(
        success=True,
        rows=[
            {
                "SK_ID_CURR": 100002,
                "prediction": "APPROVE",
                "credit_score": 710,
                "risk_band": "LOW",
                "probability": 0.12,
                "AMT_INCOME_TOTAL_CAPPED": 120000,
                "AMT_CREDIT": 250000,
                "top_features": "stable income",
            },
            {
                "SK_ID_CURR": 100038,
                "prediction": "REJECT",
                "credit_score": 420,
                "risk_band": "HIGH",
                "probability": 0.73,
                "AMT_INCOME_TOTAL_CAPPED": 67500,
                "AMT_CREDIT": 450000,
                "top_features": "high debt-to-income ratio",
            },
        ],
        row_count=2,
        columns=["SK_ID_CURR", "prediction"],
        error_message=None,
        sql_executed="SELECT * FROM applicants WHERE SK_ID_CURR IN (100002, 100038);",
    )

    text = generate_response(parsed, result, parsed.raw_query, _BrokenModel())

    assert "applicant 100002 was approved" in text.lower()
    assert "applicant 100038 was rejected" in text.lower()


def test_generate_response_falls_back_to_plain_aggregate_summary_on_model_failure():
    parsed = ParsedQuery(
        intent=IntentType.AGGREGATE_QUERY.value,
        applicant_ids=[],
        filters=[],
        columns_mentioned=["prediction"],
        raw_query="how many applicants were rejected?",
        is_general=False,
    )
    result = ExecutionResult(
        success=True,
        rows=[
            {"prediction": "REJECT", "count": 5},
            {"prediction": "APPROVE", "count": 3},
        ],
        row_count=2,
        columns=["prediction", "count"],
        error_message=None,
        sql_executed="SELECT prediction, COUNT(*) as count FROM applicants GROUP BY prediction;",
    )

    text = generate_response(parsed, result, parsed.raw_query, _BrokenModel())

    assert "5 rejected applicants" in text.lower()
    assert "3 approved applicants" in text.lower()


def test_generate_response_falls_back_to_formatted_results_on_exception():
    parsed = _sample_parsed()
    result = _sample_result(1)

    text = generate_response(parsed, result, "show high risk applicants", _BrokenModel())

    assert "SK_ID_CURR | credit_score" in text
    assert "Showing 1 result(s)" in text


def test_handle_general_question_returns_graceful_message_on_failure():
    text = handle_general_question("What is a credit score?", _BrokenModel())

    assert "credit scoring questions" in text
