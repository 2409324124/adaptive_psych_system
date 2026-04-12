from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from services import SessionStore


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
SESSION_STORE = SessionStore(
    backend=os.getenv("CAT_PSYCH_SESSION_BACKEND", "memory"),
    ttl_seconds=int(os.getenv("CAT_PSYCH_SESSION_TTL_SECONDS", "7200")),
    storage_dir=Path(os.getenv("CAT_PSYCH_SESSION_DIR", ROOT / "data" / "sessions")),
)

app = FastAPI(title="CAT-Psych API", version="0.1.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class CreateSessionRequest(BaseModel):
    scoring_model: str = Field(default="binary_2pl", pattern="^(binary_2pl|grm)$")
    max_items: int = Field(default=12, ge=1, le=50)
    min_items: int = Field(default=8, ge=1, le=50)
    device: str | None = None
    param_mode: str | None = Field(default=None, pattern="^(legacy|keyed)$")
    param_path: str | None = None
    coverage_min_per_dimension: int = Field(default=2, ge=0, le=10)
    stop_mean_standard_error: float = Field(default=0.85, gt=0.0, le=5.0)


class ResponseRequest(BaseModel):
    item_id: str
    response: int = Field(ge=1, le=5)


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


@app.get("/sessions/{session_id}/result")
def result(session_id: str) -> dict[str, object]:
    return get_session(session_id).result()


@app.get("/sessions/{session_id}/export")
def export_result(session_id: str) -> JSONResponse:
    session = get_session(session_id)
    headers = {"Content-Disposition": f'attachment; filename="cat-psych-{session_id}.json"'}
    return JSONResponse(content=session.result(), headers=headers)


@app.post("/sessions/{session_id}/restart")
def restart_session(session_id: str) -> dict[str, object]:
    session = SESSION_STORE.restart_session(session_id)
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
