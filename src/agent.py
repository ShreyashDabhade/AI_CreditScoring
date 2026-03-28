import os
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List, Optional

# LangChain imports (may raise at runtime if package not installed)
try:
    from langchain.agents import Tool, initialize_agent, AgentType
    from langchain.chat_models import ChatGoogleGenerativeAI
except Exception:
    Tool = None
    initialize_agent = None
    AgentType = None
    ChatGoogleGenerativeAI = None


def _safe_json_serializable(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


def run_agent(user_query: str, parsed_intent: Dict[str, Any], db_path: str, column_names: List[str]) -> Dict[str, Any]:
    # Validate environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"status": "error", "query_used": None, "results": [], "explanation": None, "error_message": "GEMINI_API_KEY not set"}

    # Import local modules lazily to avoid import-time failures
    try:
        from src import sql_generator, sql_validator, query_engine, shap_engine
    except Exception as e:
        return {"status": "error", "query_used": None, "results": [], "explanation": None, "error_message": "Internal modules unavailable"}

    # Create LangChain tools (stateless wrappers)
    tools = []
    if Tool is not None:
        def _gen_sql(text: str) -> str:
            try:
                from src.query_parser import parse_query

                return sql_generator.generate_sql(parse_query(text), None)
            except Exception:
                return ""

        def _validate_sql(text: str) -> str:
            try:
                result = sql_validator.validate_sql(text)
                return json.dumps(
                    {
                        "valid": result.is_valid,
                        "cleaned_sql": result.cleaned_sql,
                        "error_reason": result.error_reason,
                        "warnings": result.warnings,
                    }
                )
            except Exception:
                return json.dumps({"valid": False})

        def _exec_sql(text: str) -> str:
            try:
                df = query_engine.execute_query(text, db_path)
                # convert to records if possible
                if hasattr(df, "to_dict"):
                    records = df.to_dict(orient="records")
                else:
                    records = df
                return json.dumps(_safe_json_serializable(records))
            except Exception:
                return json.dumps([])

        def _explain(text: str) -> str:
            try:
                # expect a JSON-encoded applicant record or simple text id
                try:
                    rec = json.loads(text)
                except Exception:
                    rec = text
                expl = shap_engine.generate_explanation(rec)
                return json.dumps(_safe_json_serializable(expl))
            except Exception:
                return json.dumps({})

        tools = [
            Tool(name="generate_sql", func=_gen_sql, description="Converts a natural language query to a SQL SELECT statement for the applicants table."),
            Tool(name="validate_sql", func=_validate_sql, description="Validates that a SQL query is safe and read-only before execution."),
            Tool(name="execute_query", func=_exec_sql, description="Executes a validated SQL query and returns matching applicant records."),
            Tool(name="explain_results", func=_explain, description="Given applicant records, generates SHAP-based explanations for predictions."),
        ]

    # Initialize LLM
    llm = None
    try:
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("LangChain ChatGoogleGenerativeAI unavailable")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    except Exception:
        llm = None

    # Build system prompt (agent instructions)
    system_prompt = (
        "You are a credit analyst AI. Answer queries about applicants using the tools. "
        "The applicants table contains applicant demographic and model outputs; only use the provided tools. "
        "Always validate SQL before executing. If explanation is needed, use the explain_results tool."
    )

    # Try to create an agent (best-effort). If langchain isn't available, we still proceed with deterministic pipeline below.
    agent_exec = None
    if initialize_agent is not None and llm is not None:
        try:
            agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
            agent_exec = agent
        except Exception:
            agent_exec = None

    # Optionally run the agent in background for observability, but do not rely on its output.
    if agent_exec is not None:
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(agent_exec.run, user_query)
                try:
                    _ = fut.result(timeout=30)
                except TimeoutError:
                    # swallow timeout; proceed with deterministic pipeline
                    pass
                except Exception:
                    pass
        except Exception:
            pass

    # Deterministic pipeline (generate -> validate -> execute -> explain)
    try:
        from src.query_parser import parse_query

        sql = sql_generator.generate_sql(parse_query(user_query), None)
        if not sql or not isinstance(sql, str):
            return {"status": "error", "query_used": None, "results": [], "explanation": None, "error_message": "Could not generate SQL"}

        validation = None
        try:
            validation = sql_validator.validate_sql(sql)
        except Exception:
            validation = None

        if validation is None or not validation.is_valid:
            return {"status": "error", "query_used": sql, "results": [], "explanation": None, "error_message": "Generated SQL is not permitted"}

        sql = validation.cleaned_sql

        # Execute
        try:
            df = query_engine.execute_query(sql, db_path)
            # Convert DataFrame-like to list[dict]
            if hasattr(df, "to_dict"):
                records = df.to_dict(orient="records")
            elif isinstance(df, list):
                records = df
            else:
                # try to coerce
                records = _safe_json_serializable(df)
                if isinstance(records, dict):
                    records = [records]
        except Exception:
            return {"status": "error", "query_used": sql, "results": [], "explanation": None, "error_message": "Query execution failed"}

        # Normalize records to JSON-serializable
        try:
            records = _safe_json_serializable(records)
            if not isinstance(records, list):
                records = [records]
        except Exception:
            records = []

        explanation = None
        try:
            if parsed_intent.get("needs_explanation"):
                if parsed_intent.get("is_comparison"):
                    # expect shap_engine.compare_applicants to accept list of records
                    explanation = shap_engine.compare_applicants(records)
                else:
                    # generate explanation per applicant and return list
                    expls = []
                    for rec in records:
                        try:
                            expl = shap_engine.generate_explanation(rec)
                            expls.append(_safe_json_serializable(expl))
                        except Exception:
                            expls.append(None)
                    explanation = expls
        except Exception:
            explanation = None

        return {"status": "success", "query_used": sql, "results": records, "explanation": explanation, "error_message": None}

    except Exception:
        return {"status": "error", "query_used": None, "results": [], "explanation": None, "error_message": "Agent failed"}
