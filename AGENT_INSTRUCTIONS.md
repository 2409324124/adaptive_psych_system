# Codex CLI Handoff

## Project Location

- Local path: `/home/miku/projects/IPIP`
- Remote: `git@github.com:2409324124/adaptive_psych_system.git`
- Main deployment branch: `deploy/main-cutover-20260427`
- Upstream application baseline: `origin/main`

## Deployment Intent

- This branch is the GitHub-first deployment path for the legacy CAT-Psych web UI.
- `GET /` must serve the questionnaire homepage from `web/index.html`.
- Deployment should support a staged validation on port `8001` before cutting over
  to port `8000`.
- Production verification on `2026-04-27` succeeded with the prebuilt image path,
  not a remote rebuild.

## Runtime Shape

- FastAPI app entrypoint: `api.app:app`
- Docker service: `cat-psych`
- Container port: `8000`
- Default published port: `${PORT:-8000}:8000`
- Homepage route: `GET /`
- Health route: `GET /health`
- Static assets: mounted from `web/`
- Session data bind mount: `./data:/app/data`

## Environment

- Create `.env` from `.env.example`.
- The deployment `.env` must be a pure dotenv file. Do not leave shell commands,
  `docker run ...`, or other non-`KEY=VALUE` lines in it.
- Required external analysis variables when using DeepSeek:
  - `DEEPSEEK_API_KEY`
  - `DEEPSEEK_BASE_URL`
  - `DEEPSEEK_MODEL`
- Optional deployment variables:
  - `PORT`
  - `IMAGE_NAME`

## Recommended Rollout

1. Prepare `~/adaptive_psych_main` on the server.
2. Either:
   - pull this branch and build on the server, or
   - load the pre-exported image archive and set `IMAGE_NAME` to that tag.
3. Set `PORT=8001` and start the stack for validation.
4. Verify `GET /`, `GET /health`, and key CAT session flows.
5. Stop the validation stack, switch `PORT=8000`, and bring it up again.
6. If Cloudflare tunnel traffic still goes through the legacy
   `adaptive_psych_system_default` network, reconnect the new container there with
   the alias `app` so `cloudflared` can still resolve `http://app:8000`.

## Useful Commands

```bash
docker compose up -d --build
PORT=8001 docker compose up -d --build
curl http://127.0.0.1:${PORT:-8000}/
curl http://127.0.0.1:${PORT:-8000}/health
```

Use the preloaded image instead of a rebuild:

```bash
IMAGE_NAME=adaptive-psych-main:test PORT=8001 docker compose up -d --no-build
```

Reconnect the new container to the legacy `cloudflared` network alias when needed:

```bash
docker network connect --alias app adaptive_psych_system_default adaptive_psych_main-cat-psych-1
```
