from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from engine.constants import DISCLAIMER
from engine.database import SessionLocal, UserSessionRecord
from llm.deepseek_client import analyze_personality
from services import SessionStore


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data"
LOGGER = logging.getLogger(__name__)
@dataclass
class PersistLockEntry:
    lock: threading.Lock
    users: int = 0


_PERSIST_LOCKS: dict[str, PersistLockEntry] = {}
_PERSIST_LOCKS_GUARD = threading.Lock()

SESSION_STORE = SessionStore(
    backend=os.getenv("CAT_PSYCH_SESSION_BACKEND", "memory"),
    ttl_seconds=int(os.getenv("CAT_PSYCH_SESSION_TTL_SECONDS", "7200")),
    storage_dir=Path(os.getenv("CAT_PSYCH_SESSION_DIR", ROOT / "data" / "sessions")),
)

app = FastAPI(title="CAT-Psych API", version="0.2.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class CreateSessionRequest(BaseModel):
    scoring_model: str = Field(default="binary_2pl", pattern="^(binary_2pl|grm)$")
    max_items: int = Field(default=30, ge=1, le=50)
    min_items: int = Field(default=5, ge=1, le=50)
    device: str | None = None
    param_mode: str | None = Field(default="keyed", pattern="^(legacy|keyed)$")
    param_path: str | None = None
    coverage_min_per_dimension: int = Field(default=2, ge=0, le=10)
    stop_mean_standard_error: float = Field(default=0.65, gt=0.0, le=5.0)
    stop_stability_score: float = Field(default=0.7, ge=0.0, le=1.0)


class ResponseRequest(BaseModel):
    item_id: str
    response: int = Field(ge=1, le=5)


class CommentRequest(BaseModel):
    comment: str = Field(min_length=1, max_length=4000)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@lru_cache(maxsize=1)
def load_cat_mapping() -> dict[str, dict[str, str]]:
    return json.loads((DATA_DIR / "cat_mapping.json").read_text(encoding="utf-8"))


def enrich_with_cat_metadata(payload: dict[str, object], category_key: str | None, analysis: str | None) -> dict[str, object]:
    mapping = load_cat_mapping()
    category = mapping.get(category_key or "")
    return {
        **payload,
        "cat_category": category_key,
        "cat_name": category["name"] if category else None,
        "cat_image": category["image"] if category else None,
        "cat_image_position": category.get("image_position") if category else None,
        "cat_analysis": analysis,
    }


def build_persisted_snapshot(result_payload: dict[str, object]) -> dict[str, object]:
    return {
        "responses": result_payload.get("responses", {}),
        "path": result_payload.get("path", []),
        "user_comments": result_payload.get("user_comments", []),
        "comment_submitted": result_payload.get("comment_submitted", False),
        "disclaimer": result_payload.get("disclaimer"),
        "trait_estimates": result_payload.get("trait_estimates", {}),
        "irt_t_scores": result_payload.get("irt_t_scores", {}),
        "standard_errors": result_payload.get("standard_errors", {}),
        "uncertainty": result_payload.get("uncertainty", {}),
        "stability": result_payload.get("stability", {}),
        "classical_big5": result_payload.get("classical_big5", {}),
        "dimension_answer_counts": result_payload.get("dimension_answer_counts", {}),
        "interpretation": result_payload.get("interpretation", {}),
        "progress": result_payload.get("progress", {}),
        "progress_estimate": result_payload.get("progress_estimate", {}),
        "parameter_summary": {
            "param_mode": result_payload.get("param_mode"),
            "param_path": result_payload.get("param_path"),
            "key_aligned": result_payload.get("key_aligned"),
            "param_metadata": result_payload.get("param_metadata", {}),
        },
        "scoring_model": result_payload.get("scoring_model"),
        "created_at": result_payload.get("created_at"),
        "updated_at": result_payload.get("updated_at"),
        "max_items": result_payload.get("max_items"),
        "min_items": result_payload.get("min_items"),
        "coverage_min_per_dimension": result_payload.get("coverage_min_per_dimension"),
        "stop_mean_standard_error": result_payload.get("stop_mean_standard_error"),
        "stop_stability_score": result_payload.get("stop_stability_score"),
        "device": result_payload.get("device"),
    }


def combine_db_record(record: UserSessionRecord) -> dict[str, object]:
    raw = record.raw_responses or {}
    base = {
        "session_id": record.session_id,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": raw.get("updated_at"),
        "scoring_model": raw.get("scoring_model", "binary_2pl"),
        "max_items": raw.get("max_items"),
        "min_items": raw.get("min_items"),
        "coverage_min_per_dimension": raw.get("coverage_min_per_dimension"),
        "stop_mean_standard_error": raw.get("stop_mean_standard_error"),
        "stop_stability_score": raw.get("stop_stability_score"),
        "device": raw.get("device"),
        "responses": raw.get("responses", {}),
        "path": raw.get("path", []),
        "user_comments": raw.get("user_comments", []),
        "comment_submitted": raw.get("comment_submitted", False),
        "trait_estimates": raw.get("trait_estimates", {}),
        "irt_t_scores": record.ocean_scores or raw.get("irt_t_scores", {}),
        "standard_errors": raw.get("standard_errors", {}),
        "uncertainty": raw.get("uncertainty", {}),
        "stability": raw.get("stability", {}),
        "classical_big5": raw.get("classical_big5", {}),
        "dimension_answer_counts": raw.get("dimension_answer_counts", {}),
        "interpretation": raw.get("interpretation", {}),
        "progress": raw.get("progress", {}),
        "progress_estimate": raw.get("progress_estimate", {}),
        "param_mode": raw.get("parameter_summary", {}).get("param_mode"),
        "param_path": raw.get("parameter_summary", {}).get("param_path"),
        "key_aligned": raw.get("parameter_summary", {}).get("key_aligned"),
        "param_metadata": raw.get("parameter_summary", {}).get("param_metadata", {}),
        "runtime_state": "missing",
        "persistence_state": "persisted",
        "disclaimer": raw.get("disclaimer") or DISCLAIMER,
    }
    return enrich_with_cat_metadata(base, record.cat_category, record.llm_analysis)


def get_persisted_record(session_id: str, db: Session) -> UserSessionRecord | None:
    return db.get(UserSessionRecord, session_id)


@contextmanager
def persist_lock(session_id: str):
    with _PERSIST_LOCKS_GUARD:
        entry = _PERSIST_LOCKS.get(session_id)
        if entry is None:
            entry = PersistLockEntry(lock=threading.Lock())
            _PERSIST_LOCKS[session_id] = entry
        entry.users += 1

    entry.lock.acquire()
    try:
        yield
    finally:
        entry.lock.release()
        with _PERSIST_LOCKS_GUARD:
            current = _PERSIST_LOCKS.get(session_id)
            if current is entry:
                entry.users -= 1
                if entry.users == 0:
                    _PERSIST_LOCKS.pop(session_id, None)


def safe_delete_runtime_session(session_id: str) -> None:
    try:
        SESSION_STORE.delete_session(session_id)
    except Exception:
        LOGGER.warning(
            "Failed to purge runtime session after result persistence.",
            extra={"session_id": session_id},
            exc_info=True,
        )


def persist_session_result(session, db: Session) -> dict[str, object]:
    if not session.is_complete:
        return session.result()

    with persist_lock(session.session_id):
        record = get_persisted_record(session.session_id, db)
        if record is not None:
            safe_delete_runtime_session(session.session_id)
            return combine_db_record(record)

        result_payload = session.result()
        llm_result = analyze_personality(
            ocean_scores=result_payload["irt_t_scores"],
            user_comments=result_payload.get("user_comments", []),
            structured_summary=result_payload.get("interpretation", {}).get("structured_summary"),
        )
        persisted = UserSessionRecord(
            session_id=session.session_id,
            ocean_scores=result_payload["irt_t_scores"],
            cat_category=llm_result["category_key"],
            llm_analysis=llm_result["analysis"],
            raw_responses=build_persisted_snapshot(result_payload),
        )
        db.add(persisted)
        try:
            db.commit()
            db.refresh(persisted)
            safe_delete_runtime_session(session.session_id)
            return combine_db_record(persisted)
        except IntegrityError:
            db.rollback()
            record = get_persisted_record(session.session_id, db)
            if record is not None:
                safe_delete_runtime_session(session.session_id)
                return combine_db_record(record)
            raise


def get_result_payload(session_id: str, db: Session) -> dict[str, object]:
    record = get_persisted_record(session_id, db)
    if record is not None:
        return combine_db_record(record)

    try:
        session = get_session(session_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            record = get_persisted_record(session_id, db)
            if record is not None:
                return combine_db_record(record)
        raise
    if not session.is_complete:
        return session.result()
    return persist_session_result(session, db)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions")
def create_session(payload: CreateSessionRequest) -> dict[str, object]:
    session = SESSION_STORE.create_session(
        scoring_model=payload.scoring_model,
        max_items=payload.max_items,
        min_items=payload.min_items,
        device=payload.device,
        param_mode=payload.param_mode,
        param_path=payload.param_path,
        coverage_min_per_dimension=payload.coverage_min_per_dimension,
        stop_mean_standard_error=payload.stop_mean_standard_error,
        stop_stability_score=payload.stop_stability_score,
    )
    next_question_payload = session.next_question()
    SESSION_STORE.save_session(session)
    return {
        **session.summary(),
        "next_question": next_question_payload,
    }


@app.get("/sessions/{session_id}")
def session_summary(session_id: str) -> dict[str, object]:
    return get_session(session_id).summary()


@app.get("/sessions/{session_id}/next")
def next_question(session_id: str) -> dict[str, object]:
    session = get_session(session_id)
    question = session.next_question()
    SESSION_STORE.save_session(session)
    return {"session_id": session_id, "complete": question is None, "next_question": question}


@app.post("/sessions/{session_id}/responses")
def submit_response(session_id: str, payload: ResponseRequest) -> dict[str, object]:
    session = get_session(session_id)
    try:
        accepted = session.submit_response(payload.item_id, payload.response)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    next_question_payload = session.next_question()
    SESSION_STORE.save_session(session)
    return {**accepted, "next_question": next_question_payload}


@app.post("/sessions/{session_id}/comments")
def submit_comment(session_id: str, payload: CommentRequest, db: Session = Depends(get_db)) -> dict[str, object]:
    existing = get_persisted_record(session_id, db)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Result already finalized for this session.")

    session = get_session(session_id)
    if not session.is_complete:
        raise HTTPException(status_code=409, detail="Comments can only be submitted after the assessment is complete.")

    try:
        accepted = session.add_comment(payload.comment)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    SESSION_STORE.save_session(session)
    return {**accepted, "summary": session.summary()}


@app.get("/sessions/{session_id}/result")
def result(session_id: str, db: Session = Depends(get_db)) -> dict[str, object]:
    return get_result_payload(session_id, db)


@app.get("/results/{session_id}")
def persisted_result(session_id: str, db: Session = Depends(get_db)) -> dict[str, object]:
    record = get_persisted_record(session_id, db)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return combine_db_record(record)


@app.get("/sessions/{session_id}/export")
def export_result(session_id: str, db: Session = Depends(get_db)) -> JSONResponse:
    content = get_result_payload(session_id, db)
    headers = {"Content-Disposition": f'attachment; filename="cat-psych-{session_id}.json"'}
    return JSONResponse(content=content, headers=headers)


@app.post("/sessions/{session_id}/restart")
def restart_session(session_id: str, db: Session = Depends(get_db)) -> dict[str, object]:
    existing = get_persisted_record(session_id, db)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Finalized sessions cannot be restarted. Create a new session instead.")
    try:
        session = SESSION_STORE.restart_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session_id") from exc
    next_question_payload = session.next_question()
    SESSION_STORE.save_session(session)
    return {
        **session.summary(),
        "next_question": next_question_payload,
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, object]:
    SESSION_STORE.delete_session(session_id)
    return {"session_id": session_id, "deleted": True}


def get_session(session_id: str):
    try:
        return SESSION_STORE.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session_id") from exc
