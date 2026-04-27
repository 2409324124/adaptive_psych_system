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

## Verified 2026-04-27 Path

The production cutover that actually succeeded on `2026-04-27` used this path:

1. Clear low-risk system cache on the host first.
2. Upload a pure dotenv `.env` file from the local machine.
3. Upload `adaptive_psych_main_deploy_20260427.tgz`.
4. Upload `adaptive-psych-main-test-20260427.tar.gz`.
5. Unpack the deployment directory.
6. `docker load` the prebuilt image.
7. Start on `PORT=8001` with `IMAGE_NAME=adaptive-psych-main:test`.
8. Validate `/` and `/health`.
9. Recreate on `PORT=8000`.
10. Reconnect the new container to the legacy Docker network alias `app` so the
    existing `cloudflared` tunnel can still reach `http://app:8000`.

## Environment

Create `.env` from `.env.example` and fill in:

```text
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=
DEEPSEEK_MODEL=
PORT=8000
IMAGE_NAME=cat-psych:cpu
```

Important:

- The deployment `.env` must contain only dotenv-compatible lines.
- Do not include shell snippets, `docker run ...`, or pasted command history in
  this file.

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

If the existing Cloudflare tunnel still points to `http://app:8000`, reconnect
the new container to the legacy network alias instead of modifying `cloudflared`
first:

```bash
docker network connect --alias app adaptive_psych_system_default adaptive_psych_main-cat-psych-1
```

## Notes

- `./data` stores runtime session artifacts on the deployment host.
- This rollout path is intended for single-host deployment.
- `cloudflared` was intentionally left untouched during the successful cutover.
- Public `502` errors after the cutover were resolved by restoring the Docker
  network alias `app` for the new app container.
