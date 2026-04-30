#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  IPIP_BACKUP_REMOTE='gdrive-crypt:ipip-backups' scripts/backup_data.sh
  IPIP_BACKUP_REMOTE='gdrive-crypt:ipip-backups' scripts/backup_data.sh --dry-run

Environment:
  IPIP_DATA_DIR                    Data directory to back up. Default: <repo>/data
  IPIP_BACKUP_REMOTE               rclone destination, preferably a crypt remote.
  IPIP_BACKUP_PREFIX               Backup filename prefix. Default: ipip-data
  IPIP_BACKUP_ALLOW_UNENCRYPTED=1  Allow a non-crypt rclone remote.

The script never reads or prints .env. It backs up ./data/cat_psych.db,
./data/sessions, and ./data/results when that directory exists.
USAGE
}

DRY_RUN=0
case "${1:-}" in
  "")
    ;;
  "--dry-run")
    DRY_RUN=1
    ;;
  "-h" | "--help")
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${IPIP_DATA_DIR:-$PROJECT_ROOT/data}"
DB_PATH="$DATA_DIR/cat_psych.db"
SESSIONS_DIR="$DATA_DIR/sessions"
RESULTS_DIR="$DATA_DIR/results"
BACKUP_PREFIX="${IPIP_BACKUP_PREFIX:-ipip-data}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_NAME="${BACKUP_PREFIX}-${TIMESTAMP}.tar.gz"
TMP_PARENT="${TMPDIR:-/tmp}"
WORK_DIR="$(mktemp -d "$TMP_PARENT/ipip-data-backup.XXXXXXXX")"
PAYLOAD_DIR="$WORK_DIR/payload"
ARCHIVE_PATH="$WORK_DIR/$BACKUP_NAME"
CHECKSUM_PATH="$ARCHIVE_PATH.sha256"

cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

log() {
  printf '[backup_data] %s\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '[backup_data] required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

require_command python3
require_command tar
require_command sha256sum

if [[ ! -d "$DATA_DIR" ]]; then
  printf '[backup_data] data directory does not exist: %s\n' "$DATA_DIR" >&2
  exit 1
fi

if [[ ! -f "$DB_PATH" ]]; then
  printf '[backup_data] SQLite database not found: %s\n' "$DB_PATH" >&2
  exit 1
fi

mkdir -p "$PAYLOAD_DIR"
chmod 700 "$WORK_DIR" "$PAYLOAD_DIR"

log "creating consistent SQLite snapshot"
python3 - "$DB_PATH" "$PAYLOAD_DIR/cat_psych.db" <<'PY'
import sqlite3
import sys

src, dst = sys.argv[1], sys.argv[2]
source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
target = sqlite3.connect(dst)
try:
    with target:
        source.backup(target)
finally:
    target.close()
    source.close()
PY

if [[ -d "$SESSIONS_DIR" ]]; then
  log "including sessions directory"
  cp -a "$SESSIONS_DIR" "$PAYLOAD_DIR/sessions"
else
  log "sessions directory missing; including empty sessions directory"
  mkdir -p "$PAYLOAD_DIR/sessions"
fi

if [[ -d "$RESULTS_DIR" ]]; then
  log "including results directory"
  cp -a "$RESULTS_DIR" "$PAYLOAD_DIR/results"
else
  log "results directory not present; skipping"
fi

cat > "$PAYLOAD_DIR/manifest.txt" <<EOF
created_utc=$TIMESTAMP
data_dir=$DATA_DIR
included=cat_psych.db
included_sessions=$([[ -d "$SESSIONS_DIR" ]] && printf yes || printf no)
included_results=$([[ -d "$RESULTS_DIR" ]] && printf yes || printf no)
EOF

tar -C "$PAYLOAD_DIR" -czf "$ARCHIVE_PATH" .
(cd "$WORK_DIR" && sha256sum "$BACKUP_NAME" > "$CHECKSUM_PATH")
log "created archive $BACKUP_NAME"

if [[ "$DRY_RUN" == "1" ]]; then
  log "dry run: archive contents"
  tar -tzf "$ARCHIVE_PATH"
  log "dry run complete; no upload attempted"
  exit 0
fi

require_command rclone

RCLONE_REMOTE="${IPIP_BACKUP_REMOTE:-${RCLONE_REMOTE:-}}"
if [[ -z "$RCLONE_REMOTE" ]]; then
  printf '[backup_data] IPIP_BACKUP_REMOTE is required, for example: gdrive-crypt:ipip-backups\n' >&2
  exit 1
fi

if [[ "$RCLONE_REMOTE" != *:* ]]; then
  printf '[backup_data] IPIP_BACKUP_REMOTE must be an rclone remote path such as gdrive-crypt:ipip-backups\n' >&2
  exit 1
fi

REMOTE_NAME="${RCLONE_REMOTE%%:*}"
REMOTE_TYPE="$(rclone config show "${REMOTE_NAME}:" 2>/dev/null | awk -F '=' '/^[[:space:]]*type[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print $2; exit}')"
if [[ "${IPIP_BACKUP_ALLOW_UNENCRYPTED:-0}" != "1" && "$REMOTE_TYPE" != "crypt" ]]; then
  printf '[backup_data] refusing to upload sensitive data to non-crypt rclone remote: %s\n' "$REMOTE_NAME" >&2
  printf '[backup_data] create a crypt remote or set IPIP_BACKUP_ALLOW_UNENCRYPTED=1 after documenting the risk.\n' >&2
  exit 1
fi

DEST_DIR="${RCLONE_REMOTE%/}"
log "uploading archive to rclone destination"
rclone mkdir "$DEST_DIR"
rclone copyto "$ARCHIVE_PATH" "$DEST_DIR/$BACKUP_NAME"
rclone copyto "$CHECKSUM_PATH" "$DEST_DIR/$BACKUP_NAME.sha256"
log "backup uploaded: $DEST_DIR/$BACKUP_NAME"
