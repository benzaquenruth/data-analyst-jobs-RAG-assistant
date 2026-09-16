# loading data and building the search index
# This file handles data loading and index creation - everything we need before we can search

# job listings ingested from rag_jobs.csv (exported from BigQuery so anyone
# cloning the repo can run the assistant without needing our BigQuery credentials)
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from sqlitesearch import TextSearchIndex

load_dotenv()

CSV_PATH = "rag_jobs.csv"
DB_PATH = "jobs.db"

KEYWORD_FIELDS = ["Platform", "Company_Name", "City", "experience_bucket"]
TEXT_FIELDS = ["Title", "Job_Description", "experience_reasoning", "skills"]
# a real date field (not a keyword field) so the index supports >=/<= range
# filtering, not just exact-match - needed for the app's date-range filter
DATE_FIELDS = ["Date"]

# fields combined into one string per job for embedding
VECTOR_TEXT_FIELDS = [
    "Title", "Company_Name", "City", "Job_Description", "Date",
    "Platform", "experience_bucket", "experience_reasoning", "skills",
]
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIR = Path("data")
VECTOR_EMBEDDINGS_PATH = VECTOR_DIR / "vector_embeddings.npy"
VECTOR_DOCUMENTS_PATH = VECTOR_DIR / "vector_documents.json"
# OpenAI's embeddings endpoint caps how many inputs one request can hold
# (also caps total tokens per request at 300k - keep batches small enough to stay under that)
EMBEDDING_BATCH_SIZE = 150


# use load_jobs_data() to load the job listings from rag_jobs.csv
def load_jobs_data():
    df = pd.read_csv(CSV_PATH)
    # CSV has no nulls to speak of, but a job missing e.g. skills or a city
    # should end up as "" (searchable/joinable text), not the string "nan"
    df = df.fillna("")

    documents = df.to_dict(orient="records")
    print(f"Loaded {len(documents)} job listings")
    return documents


# reload the documents we already indexed, straight from jobs.dbbnu
def load_documents_from_db(db_path=DB_PATH):
    import sqlite3

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT doc_json FROM docs").fetchall()
    conn.close()

    return [json.loads(row[0]) for row in rows]


# sqlitesearch's TextSearchIndex.add() does a plain INSERT (no id_field is
# set below, so there's nothing to upsert on) - it just appends on top of
# whatever's already in jobs.db. To make it safe to re-run ingest.py (e.g.
# on every docker-compose up) without piling up duplicate rows, we track a
# hash of rag_jobs.csv inside jobs.db itself and only rebuild when it
# changes.
def _csv_hash():
    return hashlib.sha256(Path(CSV_PATH).read_bytes()).hexdigest()


def _get_stored_csv_hash(db_path=DB_PATH):
    if not Path(db_path).exists():
        return None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT value FROM ingest_meta WHERE key = 'csv_hash'"
        ).fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        # jobs.db exists but predates this tracking table
        return None
    finally:
        conn.close()


def _store_csv_hash(csv_hash, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS ingest_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO ingest_meta (key, value) VALUES ('csv_hash', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (csv_hash,),
    )
    conn.commit()
    conn.close()


# create the sqlitesearch keyword index and save it to a .db file on disk.
# Skips the rebuild if rag_jobs.csv hasn't changed since the last run.
def build_keyword_index(documents):
    csv_hash = _csv_hash()
    if Path(DB_PATH).exists() and _get_stored_csv_hash() == csv_hash:
        print("rag_jobs.csv hasn't changed, jobs.db is already up to date - skipping rebuild")
        return

    # dataset changed (or jobs.db doesn't exist / predates hash tracking) -
    # wipe any existing index first so we rebuild from scratch, not append
    for path in (DB_PATH, f"{DB_PATH}-shm", f"{DB_PATH}-wal"):
        Path(path).unlink(missing_ok=True)

    index = TextSearchIndex(
        text_fields=TEXT_FIELDS,
        keyword_fields=KEYWORD_FIELDS,
        date_fields=DATE_FIELDS,
        # Save the SQLite search database in a file called jobs.db
        db_path=DB_PATH
    )

    for doc in documents:
        index.add(doc)

    index.close()

    _store_csv_hash(csv_hash)
    print("Keyword index saved to jobs.db")


# with ~4,600 jobs to embed, we can burn through OpenAI's tokens-per-minute
# limit mid-run and get a RateLimitError back - retry with exponential
# backoff (same pattern the course covers) instead of letting the whole
# ingest crash partway through
MAX_EMBEDDING_RETRIES = 5


def embed_with_retry(client, batch):
    for attempt in range(MAX_EMBEDDING_RETRIES):
        try:
            return client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        except RateLimitError:
            if attempt == MAX_EMBEDDING_RETRIES - 1:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
            print(f"Rate limited, retrying in {wait}s...")
            time.sleep(wait)


# same idea as the jobs.db hash tracking above, but for the vector index:
# a small sidecar file next to the embeddings holding the rag_jobs.csv hash
# they were built from, so a re-run can tell whether it actually needs to
# call OpenAI again.
VECTOR_HASH_PATH = VECTOR_DIR / "csv_hash.txt"


# embed each job with OpenAI and save the vectors + matching documents to
# disk. Skips the (paid) embedding call if rag_jobs.csv hasn't changed
# since the last run.
def build_vector_index(documents):
    csv_hash = _csv_hash()
    if (
        VECTOR_EMBEDDINGS_PATH.exists()
        and VECTOR_DOCUMENTS_PATH.exists()
        and VECTOR_HASH_PATH.exists()
        and VECTOR_HASH_PATH.read_text().strip() == csv_hash
    ):
        print("rag_jobs.csv hasn't changed, vector index is already up to date - skipping rebuild")
        return

    # client is only created once we know we actually need to call OpenAI,
    # so this never requires OPENAI_API_KEY to be set on a skipped run
    client = OpenAI()

    texts = [
        " ".join(str(doc.get(field) or "") for field in VECTOR_TEXT_FIELDS)
        for doc in documents
    ]

    embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        response = embed_with_retry(client, batch)
        embeddings.extend(item.embedding for item in response.data)

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTOR_EMBEDDINGS_PATH, np.array(embeddings, dtype=np.float32))

    with open(VECTOR_DOCUMENTS_PATH, "w") as f:
        json.dump(documents, f)

    VECTOR_HASH_PATH.write_text(csv_hash)
    print(f"Vector embeddings saved to {VECTOR_EMBEDDINGS_PATH}")
    print(f"Vector documents saved to {VECTOR_DOCUMENTS_PATH}")


if __name__ == "__main__":
    import sys

    documents = load_jobs_data()
    build_keyword_index(documents)

    # --keyword-only skips build_vector_index() entirely, e.g. for
    # contributors without an OPENAI_API_KEY who just want the keyword
    # index. Not needed by docker-compose's `ingest` service anymore -
    # build_vector_index() now has its own hash check, so calling it
    # unconditionally is a free no-op whenever rag_jobs.csv is unchanged.
    if "--keyword-only" not in sys.argv:
        build_vector_index(documents)