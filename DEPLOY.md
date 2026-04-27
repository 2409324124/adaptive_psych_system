# Deployment Notes

This branch is the deployment-ready form of `main` for the CAT-Psych web UI.

## Baseline

- Deployment branch: `deploy/main-cutover-20260427`
- Application baseline: `origin/main`
- Expected homepage: `GET /` returns the CAT-Psych HTML app
- Health endpoint: `GET /health`

## Compose Behavior

`docker-compose.yml` supports both local builds and preloaded images:

- Default build path:
  - `docker compose up -d --build`
- Preloaded image path:
  - set `IMAGE_NAME` to the loaded tag
  - run `docker compose up -d --no-build`

Published host port is controlled by `PORT`, defaulting to `8000`.

## Environment

Create `.env` from `.env.example` and fill in:

```text
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=
DEEPSEEK_MODEL=
PORT=8000
IMAGE_NAME=cat-psych:cpu
```

## Server Rollout

Recommended rollout on the target host:

1. Create `~/adaptive_psych_main`
2. Check out `deploy/main-cutover-20260427`
3. Create `.env`
4. Start validation on port `8001`

```bash
PORT=8001 docker compose up -d --build
```

If using the pre-exported image:

```bash
docker load -i adaptive-psych-main-test-20260427.tar.gz
IMAGE_NAME=adaptive-psych-main:test PORT=8001 docker compose up -d --no-build
```

5. Validate:
   - `curl http://127.0.0.1:8001/`
   - `curl http://127.0.0.1:8001/health`
6. After homepage validation succeeds, switch to port `8000`

```bash
docker compose down
PORT=8000 docker compose up -d --build
```

Or with the preloaded image:

```bash
docker compose down
IMAGE_NAME=adaptive-psych-main:test PORT=8000 docker compose up -d --no-build
```

## Notes

- `./data` stores runtime session artifacts on the deployment host.
- This rollout path is intended for single-host deployment.
