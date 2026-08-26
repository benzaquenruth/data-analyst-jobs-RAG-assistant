# WHAT THIS FILE DOES
# -------------------
# This is a one-time migration script.
#
# The current vector index is stored in two local files:
#
#   1. data/vector_documents.json
#      Contains the job documents in the order they were embedded.
#
#   2. data/vector_embeddings.npy
#      Contains the numerical vector for each document, in the same order.
#
# This script pairs every job with its corresponding embedding and
# uploads the vectors to:
#
#   massive-bliss-481811-d8.rag_indexes.job_embeddings
#
# It does NOT create new embeddings and does NOT call OpenAI.
#
# HOW TO RUN:
#   uv run python upload_embeddings_to_bigquery.py


import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google.cloud import bigquery


# Paths to the two existing vector-index files.
DOCUMENTS_PATH = Path("data/vector_documents.json")
EMBEDDINGS_PATH = Path("data/vector_embeddings.npy")


# The BigQuery table where the embeddings will be stored.
TABLE_ID = (
    "massive-bliss-481811-d8."
    "rag_indexes.job_embeddings"
)


# The OpenAI model originally used to create these embeddings.
EMBEDDING_MODEL = "text-embedding-3-small"


# Load the environment variables stored in .env.
#
# This includes GOOGLE_APPLICATION_CREDENTIALS, which tells the
# BigQuery client where to find the service-account JSON key.
load_dotenv()


# Load the job documents from vector_documents.json.
#
# Each item represents one job posting and includes fields such as
# Link, Date, Title and Job_Description.
with DOCUMENTS_PATH.open(encoding="utf-8") as file:
    documents = json.load(file)


# Load the existing numerical embeddings.
embeddings = np.load(EMBEDDINGS_PATH)


# The number of documents and embeddings must be identical because
# they correspond according to their position:
#
#   documents[0] belongs to embeddings[0]
#   documents[1] belongs to embeddings[1]
#   and so on.
#
# Stop the script if they do not match, because otherwise embeddings
# could be connected to the wrong jobs.
if len(documents) != len(embeddings):
    raise ValueError(
        f"{len(documents)} documents but {len(embeddings)} embeddings"
    )


# Some job Links appear more than once because the same posting was
# collected on different dates.
#
# This dictionary will keep only the newest version of each Link so
# the live RAG system does not return the same job multiple times.
latest_jobs = {}


# Use one migration timestamp for all the uploaded rows.
migration_time = datetime.now(timezone.utc).isoformat()


# Pair every job document with the embedding in the same position.
for document, embedding in zip(documents, embeddings):

    # Link connects the embedding to the complete job information
    # stored in clean_jobs.
    link = str(document.get("Link") or "").strip()

    # Date is used temporarily to select the newest version when the
    # same Link appears more than once.
    job_date = str(document.get("Date") or "")

    # Every document must have a Link because it is our connection key.
    if not link:
        raise ValueError("A document is missing its Link")

    # Create one potential row for the BigQuery embeddings table.
    row = {
        "Link": link,

        # NumPy values must be converted into normal Python floats
        # before BigQuery can store them as ARRAY<FLOAT64>.
        "embedding": embedding.astype(float).tolist(),

        # Save which OpenAI model originally created the vector.
        "embedding_model": EMBEDDING_MODEL,

        # Save when the vector was transferred to BigQuery.
        "embedded_at": migration_time,

        # This temporary value is used only to compare duplicated Links.
        # It will not be uploaded to BigQuery.
        "_job_date": job_date,
    }

    # Check whether we already found another row with this Link.
    previous = latest_jobs.get(link)

    # Keep this row when:
    #   - the Link has not appeared before, or
    #   - this version has a newer Date.
    if previous is None or job_date > previous["_job_date"]:
        latest_jobs[link] = row


# Convert the unique jobs dictionary into a list of rows.
rows = list(latest_jobs.values())


# _job_date was needed only for duplicate comparison.
#
# Remove it because it is not a column in job_embeddings and the real
# job Date remains available in clean_jobs.
for row in rows:
    del row["_job_date"]


# Create the authenticated BigQuery client.
#
# It automatically uses the GOOGLE_APPLICATION_CREDENTIALS value
# loaded from .env.
client = bigquery.Client()


# Define the BigQuery schema and upload behavior.
job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("Link", "STRING"),

        # A REPEATED FLOAT64 field is ARRAY<FLOAT64> in BigQuery.
        bigquery.SchemaField(
            "embedding",
            "FLOAT64",
            mode="REPEATED",
        ),

        bigquery.SchemaField("embedding_model", "STRING"),
        bigquery.SchemaField("embedded_at", "TIMESTAMP"),
    ],

    # The temporary upload file uses newline-delimited JSON.
    source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,

    # Upload only when the table is empty.
    #
    # This prevents the script from accidentally replacing existing
    # embeddings if it is executed again.
    write_disposition=bigquery.WriteDisposition.WRITE_EMPTY,
)


# Create a temporary newline-delimited JSON file for the upload.
#
# This is safer for thousands of large vectors than sending every row
# separately through streaming inserts.
with tempfile.NamedTemporaryFile(mode="w+b", suffix=".ndjson") as file:

    # Write each BigQuery row as one JSON line.
    for row in rows:
        json_line = json.dumps(row) + "\n"
        file.write(json_line.encode("utf-8"))

    # Make sure everything is written, then return to the beginning
    # before BigQuery reads the file.
    file.flush()
    file.seek(0)

    # Upload the file and wait until BigQuery finishes processing it.
    client.load_table_from_file(
        file,
        TABLE_ID,
        job_config=job_config,
    ).result()


# Show confirmation after a successful upload.
print(f"Uploaded {len(rows)} embeddings to {TABLE_ID}")