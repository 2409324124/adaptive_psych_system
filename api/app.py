from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from services import AssessmentSession


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
SESSIONS: dict[str, AssessmentSession] = {}

app = FastAPI(title="CAT-Psych API", version="0.1.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class CreateSessionRequest(BaseModel):
    scoring_model: str = Field(default="binary_2pl", pattern="^(binary_2pl|grm)$")
    max_items: int = Field(default=12, ge=1, le=50)
    device: str | None = None


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
    session = AssessmentSession(
        scoring_model=payload.scoring_model,
        max_items=payload.max_items,
        device=payload.device,
    )
    SESSIONS[session.session_id] = session
    return {
        "session_id": session.session_id,
        "progress": session.progress(),
        "next_question": session.next_question(),
    }


@app.get("/sessions/{session_id}/next")
def next_question(session_id: str) -> dict[str, object]:
    session = get_session(session_id)
    question = session.next_question()
    return {"session_id": session_id, "complete": question is None, "next_question": question}


@app.post("/sessions/{session_id}/responses")
def submit_response(session_id: str, payload: ResponseRequest) -> dict[str, object]:
    session = get_session(session_id)
    try:
        accepted = session.submit_response(payload.item_id, payload.response)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {**accepted, "next_question": session.next_question()}


@app.get("/sessions/{session_id}/result")
def result(session_id: str) -> dict[str, object]:
    return get_session(session_id).result()


def get_session(session_id: str) -> AssessmentSession:
    try:
        return SESSIONS[session_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session_id") from exc
