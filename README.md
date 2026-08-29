

# Data Analyst Israel Jobs RAG Assistant

A RAG (Retrieval-Augmented Generation) assistant that helps you find and learn about data analyst jobs in Israel. Built as the final project for the [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) course.

:rocket: Live app: https://data-analyst-job-seeker-rag-assistant.streamlit.app/


## What is this about? 🤔

Curious what's happening in the Israeli data analyst job market? 👀  
This assistant searches **real job postings from LinkedIn and Indeed**, collected daily through an automated [data pipeline](https://github.com/benzaquenruth/data_analyst_job_seeker_automation).

Ask about roles, skills, locations, seniority, or what jobs fit your background 🚀 
📊 The data is automatically updated every day at 10:00 AM! 
<a id="live-app-diagram"></a>
**It's a job-matching assistant, not a statistics tool!** it won't answer dataset-wide questions like *"How many jobs are open in Tel Aviv?"*


<img width="1536" height="1024" alt="ChatGPT Image Aug 29, 2026, 02_58_15 AM" src="https://github.com/user-attachments/assets/51e27c2e-d882-494a-80a6-bd013ed56cde" />
<br> <br>

**Good questions to ask:**
- "Show me examples of positions someone with data engineering knowledge could apply for"
- "What jobs would fit someone with experience in BigQuery, ETL pipelines, and data integration?"
- "Best matches for a data / business analyst background"
- "Best maches for someone with SQL and python skills"
- "For financial analyst-related roles, what skills, tools, and experience are employers looking for?"

## Live app vs. running it locally 🔀


**🌐 Live app (Streamlit Cloud)**

🔄 Job postings are collected daily by the Job Seeker Automation and stored in BigQuery.\
🧠 n8n automatically creates embeddings for new jobs, while BigQuery powers both keyword and vector search.

[Take a look at the diagram ⬆️](#live-app-diagram)


**💻 Local / Docker**

📊 Use the data set inside this repo.\
📅 The current dataset in repo covers jobs from **Feb 23, 2026 to Aug 6, 2026**.

<img width="1672" height="941" alt="ChatGPT Image Aug 13, 2026, 02_55_34 AM" src="https://github.com/user-attachments/assets/8e0b3892-6505-41e7-b983-c70c68e83a55" />

```
Clone GitHub repository
        ↓
Gets monitoring.db
with your existing sample conversations
        ↓
Run Docker
        ↓
Docker uses that local monitoring.db
        ↓
User asks new questions
        ↓
New conversations + feedback
are added to THEIR local copy
```

## How to run it locally 💻

The easiest way to run this project by yourself is with Docker Compose.

### Prerequisites 🧰
- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker and Docker Compose
- An OpenAI API key

1. Clone the repo and go into it:
   ```
   git clone https://github.com/benzaquenruth/data-analyst-israel-jobs-assistant
   cd data-analyst-israel-jobs-assistant
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-key-here
   ```

3. Start the app and the dashboard:
   ```
   docker-compose up
   ```
   The keyword index (`jobs.db`) and vector index are already included in the repo, built from `rag_jobs.csv`, so this works out of the box at no cost. If you ever edit `rag_jobs.csv` yourself, the next `docker-compose up` rebuilds both automatically — the vector index rebuild calls the OpenAI embeddings API (a small cost); otherwise it's a free, fast no-op.

The assistant runs at http://localhost:8501, and the monitoring dashboard at http://localhost:8502.

## How it works

`ingest.py` reads ~4,600 real job postings from `rag_jobs.csv` and builds two search indexes:
- a **keyword index** (`sqlitesearch.TextSearchIndex`, saved to `jobs.db`)
- a **vector index** (OpenAI `text-embedding-3-small` embeddings, saved to `data/`)

When you ask a question, the assistant (`rag_helper.py`) runs both searches and combines their results with reciprocal rank fusion (hybrid search), then passes the best-matching job postings to an LLM, which writes the final answer.

## From where the data comes from?

I extracted the job postings myself from the internet and processed them
with a pipeline I built. This pipeline is part of a bigger project, and
you're welcome to take a look at it here: [data_analyst_job_seeker_automation](https://github.com/benzaquenruth/data_analyst_job_seeker_automation).

[`rag_jobs.csv`](rag_jobs.csv) is the full dataset. If you want to see what
the data looks like, check [`rag_jobs_sample.csv`](rag_jobs_sample.csv),
which has a sample of 100 rows (job descriptions truncated for readability)
and renders as a table right here on GitHub.

## Evaluation

Tested on 25 ground-truth question/job pairs (`data/ground_truth.csv`).

### 🔍 Retrieval: keyword vs. hybrid search

| Search method | boost_dict | hit_rate | mrr |
|---|---|---|---|
| Keyword only | `{"Title": 3.0, "skills": 0.5}` | 0.28 | 0.159 |
| Keyword only (production weights) | `{"skills": 4.0, "Title": 3.0, "Job_Description": 3.0}` | 0.28 | 0.159 |
| **Hybrid (keyword + vector)** | `{"skills": 4.0, "Title": 3.0, "Job_Description": 3.0}` | **0.76** | **0.435** |

Keyword search alone struggles with paraphrased questions — it only found the right job 28% of the time. Adding vector search nearly triples both metrics, so hybrid search is what the app uses in production.

*Small sample (25 questions, 5 listings) — a clear signal, not a final benchmark.*

### ⚖️ Answer quality: LLM-as-a-judge

| boost_dict | good answers |
|---|---|
| `{"Title": 3.0, "skills": 0.5}` | 19/25 |
| **`{"skills": 4.0, "Title": 3.0, "Job_Description": 3.0}`** | **23/25** |

Same story at the answer level: the tuned weights produced more good answers (23/25 vs 19/25), confirming the production setup. Full details in [`04-evaluation-notebook.ipynb`](04-evaluation-notebook.ipynb) and the `data/` folder.

## Monitoring

Every question asked in the app is logged to [`monitoring.db`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/monitoring.db), along with:
- the LLM's response time, token usage, and cost
- an LLM-as-a-judge relevance score for the answer
- optional 👍/👎 feedback from the user

The dashboard ([`dashboard.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/tree/main/pages)) reads this data and shows cost, response time, and token usage over time, the judge's relevance scores, user feedback counts, and a list of recent conversations.

## What's in this repo 📁

**[`data/`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/tree/main/data)** — data generated during evaluation, plus the vector search index:
- [`ground_truth.csv`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/data/ground_truth.csv) — test questions + correct job
- [`rag_answers.csv`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/data/rag_answers.csv) — assistant's answers during evaluation
- [`rag_evaluations.csv`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/data/rag_evaluations.csv) — judge scores for those answers
- [`csv_hash.txt`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/data/csv_hash.txt) — checksum, detects when to rebuild the vector index
- [`vector_embeddings.npy`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/data/vector_embeddings.npy) — job listing embeddings (vector search)
- [`vector_documents.json`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/data/vector_documents.json) — job listings text used for vector search

**[`pages/`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/tree/main/pages)** — extra Streamlit pages (a separate window inside the app):
- [`1_Monitoring_Dashboard.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/pages/1_Monitoring_Dashboard.py) — the monitoring dashboard

**Running the app:**
- [`ingest.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/ingest.py) — builds the keyword + vector search indexes
- [`rag_helper.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/rag_helper.py) — search + RAG logic
- [`assistant.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/assistant.py) — builds the ready-to-use assistant
- [`metrics.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/metrics.py) — tracks cost, time, tokens per call
- [`judge.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/judge.py) — LLM judge, grades answer relevance
- [`app.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/app.py) — the Streamlit app

**Monitoring backend:**
- [`bigquery_client.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/bigquery_client.py) — connects to BigQuery (live app)
- [`db_init.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/db_init.py) — creates `monitoring.db` tables (local)
- [`db_save.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/db_save.py) — saves each conversation
- [`db_feedback.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/db_feedback.py) — saves 👍/👎 and judge feedback
- [`db_query.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/db_query.py) — reads data for the dashboard

**Dataset:**
- [`rag_jobs.csv`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/rag_jobs.csv) — full job listings dataset
- [`rag_jobs_sample.csv`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/rag_jobs_sample.csv) — small 100-row preview
- [`jobs.db`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/jobs.db) — keyword search index, built from `rag_jobs.csv`

**Evaluation:**
- [`evaluation_utils.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/evaluation_utils.py) — shared helper functions for evaluation
- [`04-evaluation-notebook.ipynb`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/04-evaluation-notebook.ipynb) — full evaluation notebook
- [`rag-test.ipynb`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/rag-test.ipynb) — early testing notebook

**Setup / infra:**
- [`Dockerfile`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/Dockerfile) — builds the app's Docker image
- [`docker-compose.yaml`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/docker-compose.yaml) — runs the app + dashboard together
- [`pyproject.toml`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/pyproject.toml) — project dependencies
- [`uv.lock`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/uv.lock) — locked dependency versions

**Other:**
- [`monitoring.db`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/monitoring.db) — local monitoring data (SQLite)
- [`PROJECT_NOTES.md`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/PROJECT_NOTES.md) — working notes, not user-facing
- [`main.py`](https://github.com/benzaquenruth/data-analyst-jobs-RAG-assistant/blob/main/main.py) — default project file, not used

## Tech stack
- **LLM + embeddings:** OpenAI (`gpt-5.4-mini` for answers, `text-embedding-3-small` for embeddings)
- **Search:** `sqlitesearch` (keyword) + OpenAI embeddings (vector), combined via hybrid search
- **Interface & dashboard:** Streamlit
- **Storage:** SQLite (`jobs.db` for the search index, `monitoring.db` for monitoring data)
- **Dependency management:** uv
- **Containerization:** Docker / Docker Compose
