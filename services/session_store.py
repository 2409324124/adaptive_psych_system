from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .assessment_session import AssessmentSession


@dataclass
class SessionStore:
    backend: str = "memory"
    ttl_seconds: int = 7200
    storage_dir: Path | None = None

    def __post_init__(self) -> None:
        self.backend = self.backend.lower()
        if self.backend not in {"memory", "json"}:
            raise ValueError("backend must be either 'memory' or 'json'.")
        self.storage_dir = Path(self.storage_dir) if self.storage_dir is not None else Path(__file__).resolve().parents[1] / "data" / "sessions"
        if self.backend == "json":
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, AssessmentSession] = {}

    def create_session(
        self,
        *,
        scoring_model: str,
        max_items: int,
        min_items: int,
        device: str | None,
        param_mode: str | None,
        param_path: str | None,
        coverage_min_per_dimension: int,
        stop_mean_standard_error: float,
        stop_stability_score: float,
    ) -> AssessmentSession:
        self.cleanup_expired()
        session = AssessmentSession(
            scoring_model=scoring_model,
            max_items=max_items,
            min_items=min_items,
            device=device,
            param_mode=param_mode,
            param_path=param_path,
            coverage_min_per_dimension=coverage_min_per_dimension,
            stop_mean_standard_error=stop_mean_standard_error,
            stop_stability_score=stop_stability_score,
        )
        self.sessions[session.session_id] = session
        self._persist(session)
        return session

    def get_session(self, session_id: str) -> AssessmentSession:
        self.cleanup_expired()
        session = self.sessions.get(session_id)
        if session is None and self.backend == "json":
            session = self._load(session_id)
            if session is not None:
                self.sessions[session_id] = session
        if session is None:
            raise KeyError(session_id)
        session.touch()
        self._persist(session)
        return session

    def save_session(self, session: AssessmentSession) -> None:
        self.sessions[session.session_id] = session
        session.touch()
        self._persist(session)

    def restart_session(self, session_id: str) -> AssessmentSession:
        session = self.get_session(session_id)
        session.restart()
        self._persist(session)
        return session

    def delete_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        if self.backend == "json":
            self._delete_file(session_id)

    def cleanup_expired(self) -> None:
        now = datetime.now(UTC)
        expired_ids: list[str] = []
        for session_id, session in list(self.sessions.items()):
            updated_at = self._parse_timestamp(session.updated_at)
            if now - updated_at > timedelta(seconds=self.ttl_seconds):
                expired_ids.append(session_id)
        for session_id in expired_ids:
            self.delete_session(session_id)

        if self.backend == "json" and self.storage_dir.exists():
            for path in self.storage_dir.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    updated_at = self._parse_timestamp(str(payload["updated_at"]))
                except (OSError, json.JSONDecodeError, KeyError, ValueError):
                    continue
                if now - updated_at > timedelta(seconds=self.ttl_seconds):
                    path.unlink(missing_ok=True)

    def _persist(self, session: AssessmentSession) -> None:
        if self.backend != "json":
            return
        target = self.storage_dir / f"{session.session_id}.json"
        target.write_text(
            json.dumps(session.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self, session_id: str) -> AssessmentSession | None:
        path = self.storage_dir / f"{session_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return AssessmentSession.from_snapshot(payload)

    def _delete_file(self, session_id: str) -> None:
        path = self.storage_dir / f"{session_id}.json"
        path.unlink(missing_ok=True)

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
