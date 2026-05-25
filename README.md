# Rehal

Backend service for AI-powered quiz generation and assessment. Handles async job processing for MCQ generation using OpenAI and evaluates user quiz results with personalized feedback.

## Features

- **Auto Quiz (ATQ)** — generates multiple-choice questions from a topic/description using `o3-mini`
- **Assessment** — evaluates quiz responses, scores accuracy, and produces AI-generated advice using `gpt-4o`
- **Reassessment** — regenerates personalized advice for a previously completed assessment
- **Build Canonical Topic** — builds a canonical topic document from external sources, verifies claims, and stores embeddings
- **Credit system** — gate-keeps quiz generation and assessment behind a per-user credit balance
- **Content moderation** — screens input topics before generation

## Stack

- **FastAPI** — REST API
- **Celery + Redis** — async job queue and result backend
- **MongoDB** — document storage for quizzes and assessments
- **Tavily API** — source discovery for canonical topic generation
- **OpenAI API** — structured JSON output for generation and evaluation

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/job/submit` | Submit an async job (`autoquiz`, `assessment`, `reassessment`, `build_canonical_topic`) |
| `GET` | `/job/status/{task_id}` | Poll job status |
| `POST` | `/job/load` | Retrieve a stored document |
| `GET` | `/db-config` | Returns database and collection names |

### Submit payload shape

```json
{
  "submit_info": {
    "submit_type": "autoquiz",
    "user_id": "...",
    "doc_id": "...",
    "topic": "...",
    "description": "...",
    "num_questions": 10,
    "level": "intermediate"
  }
}
```

`submit_type` can be `autoquiz`, `assessment`, `reassessment`, or `build_canonical_topic`.

### Build canonical topic payload shape

```json
{
  "submit_info": {
    "submit_type": "build_canonical_topic",
    "user_id": "...",
    "doc_id": "...",
    "topic": "Chinese Economic Past, Present and Future",
    "min_subtopics": 4,
    "max_subtopics": 8,
    "min_sources_per_subtopic": 3,
    "max_results_per_search": 10,
    "search_max_tier": 3,
    "json_model": "gpt-4.1-mini",
    "writer_model": "gpt-4.1",
    "embedding_model": "text-embedding-3-small"
  }
}
```

The canonical topic job runs through the same async submit/status flow and stores the generated canonical document in a dedicated Mongo database and collection.

## Setup

### Requirements

- Python 3.10+
- MongoDB
- Redis

### Installation

```bash
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `REHAL_ENV_TYPE` | `DEV` or `PROD` — determines database name prefix |
| `OPENAI_API_KEY` | OpenAI API key |
| `TAVILY_API_KEY` | Tavily API key for source discovery |
| `MONGO_URI` | MongoDB connection URI |
| `REDIS_BROKER_URL` | Redis broker URL (e.g. `redis://localhost:6379/0`) |
| `REDIS_RESULT_BACKEND` | Redis result backend URL (e.g. `redis://localhost:6379/1`) |
| `REHAL_HOST` | Server host (default: `0.0.0.0`) |
| `REHAL_PORT` | Server port (default: `5500`) |
| `REHAL_RELOAD` | Enable auto-reload (default: `false`) |
| `CANONICAL_TOPIC_MONGO_DB_NAME` | Optional override for canonical topic database name |
| `CANONICAL_TOPIC_COLLECTION_NAME` | Optional override for canonical topic collection name |
| `OPENAI_CANONICAL_TOPIC_JSON_MODEL` | Model for subtopic generation, claim extraction, and clustering |
| `OPENAI_CANONICAL_TOPIC_WRITER_MODEL` | Model for final canonical document writing |
| `OPENAI_CANONICAL_TOPIC_EMBEDDING_MODEL` | Embedding model for canonical claims |
| `CANONICAL_TOPIC_MIN_SUBTOPICS` | Default minimum subtopics per topic |
| `CANONICAL_TOPIC_MAX_SUBTOPICS` | Default maximum subtopics per topic |
| `CANONICAL_TOPIC_MIN_SOURCES_PER_SUBTOPIC` | Default minimum sources required before fallback search stops |
| `CANONICAL_TOPIC_MAX_RESULTS_PER_SEARCH` | Default Tavily results per query |
| `CANONICAL_TOPIC_SEARCH_MAX_TIER` | Highest allowed source quality tier |
| `CANONICAL_TOPIC_RAW_CONTENT_LIMIT` | Max characters kept from each source body |

### Running

Start the FastAPI server:

```bash
python app_fast.py
```

Start the Celery worker:

```bash
celery -A celery_app worker --loglevel=info
```
