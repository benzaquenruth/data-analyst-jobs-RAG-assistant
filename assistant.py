# WHAT THIS FILE DOES
# This is a small "factory" file: its only job is to build a ready-to-use
# RAG assistant object, so app.py (and later, other files) don't have to
# repeat the setup steps every time. It builds a RAGWithMetrics (from
# metrics.py) instead of a plain RAGBase — same search, same answers,
# but every call also records tokens/cost/response time on
# assistant.last_call, so app.py can display and save it.

from dotenv import load_dotenv
from openai import OpenAI

from rag_helper import load_index
from metrics import RAGWithMetrics

import os 


def create_assistant():
    # Load local environment variables.
    # Streamlit Cloud provides its settings through Secrets.
    load_dotenv()

    # Streamlit Cloud already uses "bigquery".
    # Local/Docker defaults to "sqlite".
    backend = os.getenv("MONITORING_BACKEND", "sqlite")

    if backend == "bigquery":
        # Live app: use BigQuery for keyword and vector retrieval.
        from bigquery_client import get_bigquery_client

        index = None
        bigquery_client = get_bigquery_client()

    else:
        # Local/Docker: keep using jobs.db and the local vector files.
        index = load_index()
        bigquery_client = None

    return RAGWithMetrics(
        index=index,
        llm_client=OpenAI(),
        bigquery_client=bigquery_client,
    )


# This block only runs if you execute `python assistant.py` directly
# (not when app.py imports create_assistant from this file). Handy for a
# quick sanity check from the terminal without starting Streamlit.
if __name__ == "__main__":
    assistant = create_assistant()

    query = "What data analyst jobs are available in Tel Aviv?"
    answer = assistant.rag(query)
    print(answer)