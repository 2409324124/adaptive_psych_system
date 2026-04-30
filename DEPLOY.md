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

## Google Drive Data Backups

This deployment keeps runtime data on the server local disk:

- Host data directory: `./data`
- Container data directory: `/app/data`
- SQLite database: `./data/cat_psych.db`
- Session files: `./data/sessions`
- Optional result files: `./data/results`

Do not mount Google Drive as `./data` or `/app/data`. SQLite must remain on the
server local disk; Google Drive is only a backup target.

Backups are created by [`scripts/backup_data.sh`](scripts/backup_data.sh). The
script:

- checks that `./data/cat_psych.db` exists before backing up;
- creates a consistent SQLite snapshot with Python's SQLite backup API;
- includes `./data/sessions`;
- includes `./data/results` when that directory exists;
- creates timestamped files such as `ipip-data-20260430T030000Z.tar.gz`;
- uploads the archive and a `.sha256` checksum with `rclone`;
- does not read or print `.env`, API keys, or rclone secrets.

### Recommended encrypted rclone target

Psychological assessment data is sensitive. Prefer an rclone `crypt` remote on
top of Google Drive:

Install rclone if the server does not have it:

```bash
sudo apt-get update
sudo apt-get install -y rclone
```

```bash
rclone config
```

Create a normal Google Drive remote first, for example `gdrive:`. Then create a
crypt remote, for example:

```text
Name: gdrive-crypt
Type: crypt
Remote to encrypt/decrypt: gdrive:ipip-backups
Filename encryption: standard
Directory name encryption: true
```

The backup script refuses to upload to a non-crypt remote by default. If you
temporarily set `IPIP_BACKUP_ALLOW_UNENCRYPTED=1`, the archive will be uploaded
without rclone crypt encryption. That is a privacy risk and should only be used
as a short-lived emergency fallback after documenting who can access the Drive
folder.

### Manual backup

From the deployment directory:

Current production path, verified on `2026-04-30`:

```text
/home/xu2409324124/adaptive_psych_main
```

If a future host uses a different checkout path, replace this path in the
manual, cron, and restore commands before installing the schedule.

```bash
cd /home/xu2409324124/adaptive_psych_main
IPIP_BACKUP_REMOTE='gdrive-crypt:ipip-backups' ./scripts/backup_data.sh
```

Dry-run without uploading:

```bash
cd /home/xu2409324124/adaptive_psych_main
IPIP_BACKUP_REMOTE='gdrive-crypt:ipip-backups' ./scripts/backup_data.sh --dry-run
```

List uploaded backups:

```bash
rclone lsf 'gdrive-crypt:ipip-backups' --include 'ipip-data-*.tar.gz'
```

Verify an uploaded archive and checksum manually:

```bash
mkdir -p /tmp/ipip-restore-check
rclone copyto 'gdrive-crypt:ipip-backups/ipip-data-YYYYMMDDTHHMMSSZ.tar.gz' /tmp/ipip-restore-check/ipip-data.tar.gz
rclone copyto 'gdrive-crypt:ipip-backups/ipip-data-YYYYMMDDTHHMMSSZ.tar.gz.sha256' /tmp/ipip-restore-check/ipip-data.tar.gz.sha256
cd /tmp/ipip-restore-check
sha256sum -c ipip-data.tar.gz.sha256
tar -tzf ipip-data.tar.gz | head
```

### Scheduled backup every 3 days

Cron example:

```cron
0 3 */3 * * cd /home/xu2409324124/adaptive_psych_main && IPIP_BACKUP_REMOTE='gdrive-crypt:ipip-backups' ./scripts/backup_data.sh >> /var/log/ipip-data-backup.log 2>&1
```

Install with:

```bash
crontab -e
```

Make sure the cron user has access to the rclone config. If rclone was
configured as another user, either run cron as that user or set
`RCLONE_CONFIG=/path/to/rclone.conf` in the crontab. Do not store `.env` or API
keys in the crontab.

### Restore from backup

Use a maintenance window. Stop the app before replacing SQLite and session
files.

```bash
cd /home/xu2409324124/adaptive_psych_main
mkdir -p /tmp/ipip-restore
rclone copyto 'gdrive-crypt:ipip-backups/ipip-data-YYYYMMDDTHHMMSSZ.tar.gz' /tmp/ipip-restore/ipip-data.tar.gz
rclone copyto 'gdrive-crypt:ipip-backups/ipip-data-YYYYMMDDTHHMMSSZ.tar.gz.sha256' /tmp/ipip-restore/ipip-data.tar.gz.sha256
cd /tmp/ipip-restore
sha256sum -c ipip-data.tar.gz.sha256
mkdir extracted
tar -xzf ipip-data.tar.gz -C extracted
```

Before copying restored data into place, keep a local emergency copy of the
current data:

```bash
cd /home/xu2409324124/adaptive_psych_main
cp -a data "data.before-restore.$(date -u +%Y%m%dT%H%M%SZ)"
docker compose stop cat-psych
cp /tmp/ipip-restore/extracted/cat_psych.db data/cat_psych.db
rm -rf data/sessions
cp -a /tmp/ipip-restore/extracted/sessions data/sessions
if [ -d /tmp/ipip-restore/extracted/results ]; then
  rm -rf data/results
  cp -a /tmp/ipip-restore/extracted/results data/results
fi
docker compose up -d
docker compose ps
docker compose port cat-psych 8000
curl http://127.0.0.1:8000/health
```

The current production container publishes `cat-psych` on host port `8000`.
If a future deployment maps the service to another host port, use the value
shown by `docker compose port cat-psych 8000` in the final `curl` command.

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
