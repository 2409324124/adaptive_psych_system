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
  default to 100 answers, 1000 characters per field, and 4000 total
  questionnaire characters.

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

## Verified Local Drill

The current `chore/deploy-prep` branch was exercised through Docker Compose with
real HTTP requests:

- Normal questionnaire: returned `200`, produced a `submission_id`, called the
  external analysis API once, wrote SQLite data, and created a result cache.
- Duplicate submission: repeated identical submissions reused the existing
  `submission_id` and cached analysis without another external API call.
- Extreme legal input: a near-limit payload with 95 answers, one field near 1000
  characters, and total questionnaire text under 4000 characters completed
  successfully without timeout or server error.
- Over-limit input: single field over 1000 characters, total text over 4000
  characters, more than 100 answers, and empty questionnaires were rejected with
  sanitized `422` responses and did not call the external API.
- Rate limiting: questionnaire submit and analysis endpoints returned `429`
  after their configured per-window limits.
- Container restart persistence: `/app/data/app.sqlite3` and result cache files
  remained available after restarting the app container.
- External API failure: simulated upstream failure returned a controlled `502`,
  did not write a failed submission, and did not log secrets.

## Deployment Risks

- Rate limits are in memory and reset on app restart.
- SQLite is suitable only for one host and low write concurrency.
- Concurrent duplicate-submission races have not been covered.
- Disk-full behavior has not been covered.
- Slow external API responses have not been covered.
- HTTPS, reverse proxy, and authentication are not configured yet.
