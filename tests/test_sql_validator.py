from src.sql_validator import (
    ValidationResult,
    check_column_names,
    check_forbidden_keywords,
    check_table_name,
    extract_columns_from_sql,
    sanitize_sql,
    validate_sql,
)


def test_check_forbidden_keywords_detects_drop():
    assert check_forbidden_keywords("SELECT * FROM applicants; DROP TABLE applicants;") == "DROP"
    assert check_forbidden_keywords("SELECT * FROM applicants") is None


def test_check_table_name_requires_applicants_only():
    assert check_table_name("SELECT * FROM applicants")
    assert not check_table_name("SELECT * FROM loans")
    assert not check_table_name("SELECT * FROM applicants JOIN loans ON applicants.applicant_id = loans.id")


def test_extract_columns_from_sql_handles_select_where_and_order():
    sql = (
        "SELECT applicant_id, AVG(income) AS avg_income "
        "FROM applicants WHERE risk_band = 'HIGH' ORDER BY credit_score DESC"
    )

    columns = extract_columns_from_sql(sql)

    assert "applicant_id" in columns
    assert "income" in columns
    assert "risk_band" in columns
    assert "credit_score" in columns


def test_check_column_names_returns_only_invalid_columns():
    invalid = check_column_names(
        "SELECT credit_score, fake_column FROM applicants WHERE made_up = 1 ORDER BY applicant_id"
    )

    assert invalid == ["fake_column", "made_up"]


def test_sanitize_sql_strips_markdown_and_appends_semicolon():
    cleaned = sanitize_sql("```sql\nSELECT *   FROM applicants WHERE applicant_id = 12345\n```")

    assert cleaned == "SELECT * FROM applicants WHERE applicant_id = 12345;"


def test_validate_sql_returns_valid_result_with_warning_for_bad_column():
    result = validate_sql("SELECT fake_column FROM applicants WHERE applicant_id = 12345")

    assert isinstance(result, ValidationResult)
    assert result.is_valid is True
    assert result.cleaned_sql.endswith(";")
    assert result.error_reason is None
    assert result.warnings == ["Potentially invalid column names: fake_column"]


def test_validate_sql_rejects_drop_and_allows_unpacking():
    result = validate_sql("SELECT * FROM applicants; DROP TABLE applicants;")
    ok, reason = result

    assert ok is False
    assert "Forbidden SQL keyword detected: DROP" == reason
    assert bool(result) is False
