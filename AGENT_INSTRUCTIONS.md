# Codex CLI Handoff

## Project Location

- Local path: `/home/miku/projects/IPIP`
- Remote: `git@github.com:2409324124/adaptive_psych_system.git`
- Current working/deployment branch: `chore/deploy-prep`
- First deployment candidate commit: `885437f7e44b5a31e03ec9a0d2c144f0e3bd1e5c`
- Do not use `main` as the deployment baseline for now.

## Branch Policy

- Continue all deployment and small production-readiness work on `chore/deploy-prep`.
- Cloud servers should clone or checkout `chore/deploy-prep` directly.
- Do not force-push or overwrite `main`.
- Do not try to PR this branch into `main` right now; the histories are unrelated.
- If unified Git history is needed later, handle it as a separate task.

## Current Runtime Shape

- FastAPI app entrypoint: `api.app:app`
- Docker command: `uvicorn api.app:app --host 0.0.0.0 --port 8000`
- Docker Compose service: `app`
- Public port mapping: `${PORT:-8000}:8000`
- Health route: `GET /health`
- Database: SQLite
- SQLite path inside container: `/app/data/app.sqlite3`
- Persistent volume: `ipip_data:/app/data`
- Persisted data includes SQLite, `/app/data/sessions`, and `/app/data/results`.

## Environment

- Local `.env` exists and contains real API config. Do not print or commit it.
- `.env` is ignored by Git.
- Use `.env.example` as the template for deployment.
- Required external analysis variables:
  - `DEEPSEEK_API_KEY`
  - `DEEPSEEK_BASE_URL`
  - `DEEPSEEK_MODEL`

## Main API Flow

User flow:

1. User fills questionnaire.
2. Client submits to `POST /questionnaires`.
3. Server validates size/rate/duplicate rules.
4. Server calls external LLM analysis through `call_external_analysis()`.
5. Server writes SQLite row and result cache file.
6. Server returns `submission_id` and analysis text.

External API can be triggered only by:

- `POST /questionnaires`
- `POST /questionnaires/{submission_id}/analyze`, only when no cached analysis exists
- Internal function `call_external_analysis()`

## Current Protections

- Submit rate limit: default 5 requests / 60 seconds.
- Analyze rate limit: default 3 requests / 60 seconds.
- Duplicate submission window: default 300 seconds.
- Input limits:
  - `MAX_ANSWERS=100`
  - `MAX_FIELD_CHARS=1000`
  - `MAX_TOTAL_CHARS=4000`
- `422` validation responses are sanitized and do not echo full user input.
- External API call/failure logs include payload hashes only, not secrets.

## Useful Commands

```bash
cd /home/miku/projects/IPIP
git status --short --branch
docker compose up -d --build
docker compose ps
curl http://127.0.0.1:${PORT:-8000}/health
docker logs --tail=100 ipip-app-1
```

Local machine note: `.env` currently sets `PORT=8001` because port `8000` was already occupied during testing.

## Verified Local Drill

This branch has been tested with real HTTP requests through Docker Compose:

- Normal questionnaire submission returned `200`, produced `submission_id`, called external API once, wrote SQLite, and generated result cache.
- Repeated identical submissions reused the same result and did not repeat the external API call.
- Extreme legal input near configured limits completed without timeout or `500`.
- Invalid over-limit input returned sanitized `422` and did not call external API.
- Submit/analyze rate limits returned `429`.
- Container restart preserved `/app/data/app.sqlite3` and result cache files.
- Simulated external API failure returned controlled `502`, did not write a failed submission, and did not leak secrets.

## Known Risks

- Rate limits are in memory and reset on app restart.
- SQLite is suitable only for single-host, low-concurrency deployment.
- Concurrent duplicate-submission races have not been covered.
- Disk-full behavior has not been covered.
- Slow external API responses have not been covered.
- HTTPS, reverse proxy, and authentication are not configured yet.

## Do Not Do

- Do not commit `.env` or print secrets.
- Do not switch to PostgreSQL, Redis, Kubernetes, or a larger stack unless explicitly requested.
- Do not overwrite `main`.
- Do not create a new repository.
- Do not touch unrelated repos such as `/home/miku/projects/gpt-researcher`.
