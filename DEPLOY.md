# Deployment Notes

This branch is prepared for a single-host, low-concurrency Docker Compose deployment.

## Runtime

- App: FastAPI served by Uvicorn
- Database: SQLite
- SQLite path: `/app/data/app.sqlite3`
- Persistent data volume: `ipip_data:/app/data`
- Persisted runtime data: SQLite database, `/app/data/sessions`, `/app/data/results`
- Health check: `GET /health`
- Public port: `${PORT:-8000}:8000`

## Environment

Create `.env` from `.env.example` and fill in the external analysis API values:

```text
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

## High-cost API Protection

The external analysis API is called only by:

- `POST /questionnaires`
- `POST /questionnaires/{submission_id}/analyze`, only when the submission has no cached analysis
- `call_external_analysis()` inside `api/app.py`

Protections:

- Submit rate limit: `SUBMIT_LIMIT_PER_WINDOW`
- Analyze rate limit: `ANALYZE_LIMIT_PER_WINDOW`
- Rate limit window: `RATE_LIMIT_WINDOW_SECONDS`
- Duplicate submission window: `DUPLICATE_WINDOW_SECONDS`
- Input size limits: `MAX_ANSWERS`, `MAX_FIELD_CHARS`, `MAX_TOTAL_CHARS`

The limits are intentionally single-process and single-host. They are enough for
low-concurrency deployment, but should be replaced before running multiple app
instances.

## Local Production-Style Run

```bash
docker compose up -d --build
curl http://127.0.0.1:${PORT:-8000}/health
```

Submit a questionnaire:

```bash
curl -X POST http://127.0.0.1:${PORT:-8000}/questionnaires \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","answers":[{"question_id":"q1","question":"最近压力如何？","answer":"压力较大，睡眠不好，但还能工作。"}]}'
```

Re-analyze an existing submission. This returns cached analysis when available:

```bash
curl -X POST http://127.0.0.1:${PORT:-8000}/questionnaires/{submission_id}/analyze
```

## Deployment Risks

- SQLite is suitable only for one host and low write concurrency.
- Rate limits are in memory and reset on app restart.
- Run behind HTTPS and a reverse proxy before exposing it publicly.
- Keep `.env`, SQLite files, caches, and logs out of Git and Docker images.
