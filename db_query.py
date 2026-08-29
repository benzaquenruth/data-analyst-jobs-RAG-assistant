# WHAT THIS FILE DOES
# This is the "read" side of monitoring — the opposite of db_save.py
# and db_feedback.py (which write data in).
#
# These functions pull monitoring data back OUT, already shaped the way
# the monitoring dashboard needs it: recent conversations, overall stats,
# judge-relevance counts, and user thumbs up/down counts.
#
# Local / Docker:
#   -> reads from monitoring.db (SQLite)
#
# Streamlit Cloud:
#   -> reads from BigQuery, from the rag_monitoring dataset


import os
from dataclasses import dataclass

from db_init import get_db_connection
from metrics import LLMCallRecord


# Stats holds the four headline numbers shown at the top of the
# dashboard. A dataclass here is just a simple named container, same
# idea as LLMCallRecord in metrics.py.
@dataclass
class Stats:
    total: int
    avg_response_time: float
    total_cost: float
    avg_tokens: float


# row_to_record() converts one raw row from the "conversations" table
# into an LLMCallRecord — the same object shape metrics.py uses.
# This works for rows coming from both SQLite and BigQuery.
def row_to_record(row):
    return LLMCallRecord(
        model=row[3],
        instructions=row[4],
        prompt=row[5],
        answer=row[2],
        prompt_tokens=row[6],
        completion_tokens=row[7],
        total_tokens=row[8],
        response_time=row[9],
        cost=row[10],
        timestamp=row[11],
    )


# get_conversations() returns the most recent `limit` conversations,
# newest first — used for the "recent conversations" list on the dashboard.
def get_conversations(limit=10):

    backend = os.getenv("MONITORING_BACKEND", "sqlite")

    if backend == "bigquery":
        from bigquery_client import get_bigquery_client

        client = get_bigquery_client()

        query = f"""
            SELECT
                c.id, c.question, c.answer, c.model,
                c.instructions, c.prompt,
                c.prompt_tokens, c.completion_tokens, c.total_tokens,
                c.response_time, c.cost, c.timestamp,
                f.relevance, f.explanation
            FROM `massive-bliss-481811-d8.rag_monitoring.conversations` AS c
            LEFT JOIN `massive-bliss-481811-d8.rag_monitoring.feedback` AS f
                ON c.id = f.conversation_id
                AND f.source = 'judge'
            ORDER BY c.timestamp DESC
            LIMIT {int(limit)}
        """

        rows = list(client.query(query).result())

    else:
        conn = get_db_connection()
        try:
            rows = conn.execute(
                """
                SELECT
                    c.id, c.question, c.answer, c.model,
                    c.instructions, c.prompt,
                    c.prompt_tokens, c.completion_tokens, c.total_tokens,
                    c.response_time, c.cost, c.timestamp,
                    f.relevance, f.explanation
                FROM conversations AS c
                LEFT JOIN feedback AS f
                    ON c.id = f.conversation_id
                    AND f.source = 'judge'
                ORDER BY c.timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()

    results = []

    for row in rows:
        record = row_to_record(row)

        results.append({
            "id": row[0],
            "question": row[1],
            "record": record,
            "relevance": row[12],
            "explanation": row[13],
        })

    return results


# get_stats() answers: "give me the main monitoring numbers across ALL
# conversations" — shown as the summary metrics at the top of the dashboard.
def get_stats():

    backend = os.getenv("MONITORING_BACKEND", "sqlite")

    if backend == "bigquery":
        from bigquery_client import get_bigquery_client

        client = get_bigquery_client()

        query = """
            SELECT
                COUNT(*),
                AVG(response_time),
                SUM(cost),
                AVG(total_tokens)
            FROM `massive-bliss-481811-d8.rag_monitoring.conversations`
        """

        row = next(client.query(query).result())

    else:
        # -------------------------
        # LOCAL / DOCKER → SQLite
        # -------------------------

        conn = get_db_connection()
        try:
            row = conn.execute("""
                SELECT
                    COUNT(*),
                    AVG(response_time),
                    SUM(cost),
                    AVG(total_tokens)
                FROM conversations
            """).fetchone()
        finally:
            conn.close()

    return Stats(
        total=row[0],
        avg_response_time=row[1] or 0.0,
        total_cost=row[2] or 0.0,
        avg_tokens=row[3] or 0.0,
    )


# get_relevance_stats() returns how many judge verdicts fall into each
# relevance category, e.g. {"RELEVANT": 8, "PARTLY_RELEVANT": 2}.
def get_relevance_stats():

    backend = os.getenv("MONITORING_BACKEND", "sqlite")

    if backend == "bigquery":
        from bigquery_client import get_bigquery_client

        client = get_bigquery_client()

        query = """
            SELECT relevance, COUNT(*)
            FROM `massive-bliss-481811-d8.rag_monitoring.feedback`
            WHERE source = 'judge'
            GROUP BY relevance
        """

        rows = list(client.query(query).result())

    else:
        # -------------------------
        # LOCAL / DOCKER → SQLite
        # -------------------------

        conn = get_db_connection()
        try:
            rows = conn.execute("""
                SELECT relevance, COUNT(*)
                FROM feedback
                WHERE source = 'judge'
                GROUP BY relevance
            """).fetchall()
        finally:
            conn.close()

    return dict(rows)


# get_user_feedback_stats() returns (thumbs_up_count, thumbs_down_count)
# from real user feedback (score = +1 or -1).
def get_user_feedback_stats():

    backend = os.getenv("MONITORING_BACKEND", "sqlite")

    if backend == "bigquery":
        from bigquery_client import get_bigquery_client

        client = get_bigquery_client()

        query = """
            SELECT
                SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END)
            FROM `massive-bliss-481811-d8.rag_monitoring.feedback`
            WHERE source = 'user'
        """

        row = next(client.query(query).result())

    else:
        # -------------------------
        # LOCAL / DOCKER → SQLite
        # -------------------------

        conn = get_db_connection()
        try:
            row = conn.execute("""
                SELECT
                    SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END)
                FROM feedback
                WHERE source = 'user'
            """).fetchone()
        finally:
            conn.close()

    return row[0] or 0, row[1] or 0


# This lets us do a quick sanity check from the terminal:
#   uv run python db_query.py
#
# Locally, MONITORING_BACKEND is not set, so this reads monitoring.db.
if __name__ == "__main__":
    print("Stats:", get_stats())
    print("Relevance stats:", get_relevance_stats())
    print("User feedback (up, down):", get_user_feedback_stats())
    print("Recent conversations:")

    for c in get_conversations(limit=5):
        print(" -", c["id"], c["question"])