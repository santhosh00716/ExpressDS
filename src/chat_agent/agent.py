"""
Agentic Chat - LangChain + Google Gemini for natural language data queries.
Uses langchain_google_genai when available (Python 3.10+), else direct Gemini REST API.
"""

import pandas as pd
from typing import Optional
import os
import sys
import json
import urllib.request
import urllib.error


def _create_dataframe_context(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Build a text context from the DataFrame for LLM."""
    sample = df.head(max_rows).to_string()
    summary = df.describe(include="all").to_string() if len(df) > 0 else ""
    cols = ", ".join(df.columns.tolist())
    return f"Columns: {cols}\n\nShape: {df.shape[0]} rows x {df.shape[1]} columns\n\nSample (first {min(max_rows, len(df))} rows):\n{sample}\n\nSummary:\n{summary}"


def _call_gemini_rest_api(api_key: str, model: str, prompt: str) -> str:
    """Call Gemini API via REST (no langchain_google_genai required)."""
    fallback_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
    models_to_try = [model] + [m for m in fallback_models if m != model]

    last_error = None
    for m in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0},
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            break
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 404:
                continue
            err_body = e.read().decode() if e.fp else str(e)
            raise RuntimeError(f"Gemini API error: {e.code} - {err_body}") from e
    else:
        err_body = last_error.read().decode() if last_error and last_error.fp else str(last_error)
        raise RuntimeError(f"Gemini API: model not found. Tried: {models_to_try}. Error: {err_body}")
    text = ""
    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            text += part.get("text", "")
    return text.strip() or "No response from Gemini."


def _create_fallback_agent(df: pd.DataFrame, api_key: str, model_name: str):
    """
    Fallback: use Gemini REST API with dataframe context (no langchain required).
    """
    context = _create_dataframe_context(df)

    def ask(question: str) -> str:
        system = (
            "You are a helpful data analyst. Answer questions about the dataset below. "
            "Use the data context to provide accurate answers. If the answer cannot be "
            "determined from the context, say so. Be concise."
        )
        prompt = f"{system}\n\nData context:\n{context}\n\nQuestion: {question}"
        return _call_gemini_rest_api(api_key, model_name, prompt)

    class FallbackAgent:
        def invoke(self, inp):
            q = inp.get("input", inp) if isinstance(inp, dict) else str(inp)
            return {"output": ask(q)}

    return FallbackAgent()


def _get_langchain_gemini_llm(api_key: str, model_name: str):
    """Create Gemini LLM via langchain_google_genai (requires Python 3.10+)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
    )


def create_data_agent(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
) -> Optional[object]:
    """
    Create a data chat agent using Google Gemini.
    Uses langchain_google_genai if installed (Python 3.10+), else direct REST API.
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None

    # Prefer REST API fallback when langchain_google_genai is not installed (e.g. Python 3.8)
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        return _create_fallback_agent(df, key, model_name)

    # Python 3.8: pandas agent uses PythonAstREPLTool which needs 3.9+
    if sys.version_info < (3, 9):
        return _create_fallback_agent(df, key, model_name)

    try:
        from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

        llm = _get_langchain_gemini_llm(key, model_name)
        agent = create_pandas_dataframe_agent(
            llm, df, verbose=False, allow_dangerous_code=True
        )
        return agent
    except Exception as e:
        raise RuntimeError(f"Failed to create data agent: {e}") from e


def chat_with_data(
    agent: object,
    question: str,
) -> str:
    """
    Run a natural language query against the data agent.
    """
    if agent is None:
        return "Error: No agent available. Please set your Gemini API key in the sidebar."

    try:
        response = agent.invoke({"input": question})
        if isinstance(response, dict) and "output" in response:
            return str(response["output"])
        return str(response)
    except Exception as e:
        err = str(e)
        if "401" in err or "403" in err or "invalid" in err.lower() or "API key" in err:
            return (
                "**Invalid Gemini API Key.**\n\n"
                "Please check:\n"
                "1. Enter your key in the sidebar\n"
                "2. Get a free key at https://aistudio.google.com/apikey\n"
                "3. Ensure no extra spaces and the key is valid"
            )
        return f"Error: {err}"
