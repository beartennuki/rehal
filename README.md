# Rehal

Backend service for AI-powered quiz generation and assessment. Handles async job processing for MCQ generation using OpenAI and evaluates user quiz results with personalized feedback.

## Features

- **Auto Quiz (ATQ)** — generates multiple-choice questions from a topic/description using `o3-mini`
- **Assessment** — evaluates quiz responses, scores accuracy, and produces AI-generated advice using `gpt-4o`
- **Reassessment** — regenerates personalized advice for a previously completed assessment
- **Credit system** — gate-keeps quiz generation and assessment behind a per-user credit balance
- **Content moderation** — screens input topics before generation

## Stack

- **FastAPI** — REST API
- **Celery + Redis** — async job queue and result backend
- **MongoDB** — document storage for quizzes and assessments
- **OpenAI API** — structured JSON output for generation and evaluation

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/job/submit` | Submit an async job (`autoquiz`, `assessment`, `reassessment`) |
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

`submit_type` can be `autoquiz`, `assessment`, or `reassessment`.

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
| `MONGO_URI` | MongoDB connection URI |
| `REDIS_BROKER_URL` | Redis broker URL (e.g. `redis://localhost:6379/0`) |
| `REDIS_RESULT_BACKEND` | Redis result backend URL (e.g. `redis://localhost:6379/1`) |
| `REHAL_HOST` | Server host (default: `0.0.0.0`) |
| `REHAL_PORT` | Server port (default: `5500`) |
| `REHAL_RELOAD` | Enable auto-reload (default: `false`) |

### Running

Start the FastAPI server:

```bash
python app_fast.py
```

Start the Celery worker:

```bash
celery -A celery_app worker --loglevel=info
```
