from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator


DATA_DIR = Path(os.getenv("APP_DATA_DIR", "/app/data"))
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(DATA_DIR / "app.sqlite3")))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(DATA_DIR / "results")))

MAX_ANSWERS = int(os.getenv("MAX_ANSWERS", "100"))
MAX_FIELD_CHARS = int(os.getenv("MAX_FIELD_CHARS", "2000"))
MAX_TOTAL_CHARS = int(os.getenv("MAX_TOTAL_CHARS", "8000"))
DUPLICATE_WINDOW_SECONDS = int(os.getenv("DUPLICATE_WINDOW_SECONDS", "300"))
SUBMIT_LIMIT_PER_WINDOW = int(os.getenv("SUBMIT_LIMIT_PER_WINDOW", "5"))
ANALYZE_LIMIT_PER_WINDOW = int(os.getenv("ANALYZE_LIMIT_PER_WINDOW", "3"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


app = FastAPI()


class Answer(BaseModel):
    question_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=MAX_FIELD_CHARS)
    answer: str = Field(min_length=1, max_length=MAX_FIELD_CHARS)


class QuestionnaireSubmission(BaseModel):
    user_id: str | None = Field(default=None, max_length=128)
    answers: list[Answer] = Field(min_length=1, max_length=MAX_ANSWERS)
    notes: str | None = Field(default=None, max_length=MAX_FIELD_CHARS)

    @field_validator("answers")
    @classmethod
    def validate_total_size(cls, answers: list[Answer]) -> list[Answer]:
        total = sum(len(a.question_id) + len(a.question) + len(a.answer) for a in answers)
        if total > MAX_TOTAL_CHARS:
            raise ValueError(f"questionnaire is too large; max {MAX_TOTAL_CHARS} characters")
        return answers


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def startup() -> None:
    ensure_storage()


@app.post("/questionnaires")
def submit_questionnaire(payload: QuestionnaireSubmission, request: Request) -> dict[str, Any]:
    client_key = client_identity(request, payload.user_id)
    enforce_rate_limit(f"submit:{client_key}", SUBMIT_LIMIT_PER_WINDOW)

    payload_json = payload.model_dump(mode="json")
    payload_hash = stable_hash(payload_json)
    duplicate = find_recent_duplicate(payload_hash)
    if duplicate:
        return {
            "submission_id": duplicate["id"],
            "duplicate": True,
            "analysis": duplicate["analysis_text"],
        }

    submission_id = uuid.uuid4().hex
    analysis = call_external_analysis(payload_json)
    now = time.time()
    with db() as conn:
        conn.execute(
            """
            INSERT INTO submissions
                (id, user_key, payload_hash, payload_json, analysis_text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                submission_id,
                client_key,
                payload_hash,
                json.dumps(payload_json, ensure_ascii=False),
                analysis,
                now,
                now,
            ),
        )
    write_result(submission_id, analysis)
    return {"submission_id": submission_id, "duplicate": False, "analysis": analysis}


@app.post("/questionnaires/{submission_id}/analyze")
def analyze_submission(submission_id: str, request: Request) -> dict[str, Any]:
    client_key = client_identity(request)
    enforce_rate_limit(f"analyze:{client_key}", ANALYZE_LIMIT_PER_WINDOW)

    with db() as conn:
        row = conn.execute(
            "SELECT payload_json, analysis_text FROM submissions WHERE id = ?",
            (submission_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="submission not found")
    if row["analysis_text"]:
        return {"submission_id": submission_id, "cached": True, "analysis": row["analysis_text"]}

    analysis = call_external_analysis(json.loads(row["payload_json"]))
    with db() as conn:
        conn.execute(
            "UPDATE submissions SET analysis_text = ?, updated_at = ? WHERE id = ?",
            (analysis, time.time(), submission_id),
        )
    write_result(submission_id, analysis)
    return {"submission_id": submission_id, "cached": False, "analysis": analysis}


def ensure_storage() -> None:
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY,
                user_key TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                analysis_text TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_submissions_hash_created ON submissions(payload_hash, created_at)"
        )


@contextmanager
def db():
    ensure_parent = SQLITE_PATH.parent
    ensure_parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def stable_hash(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def find_recent_duplicate(payload_hash: str) -> sqlite3.Row | None:
    cutoff = time.time() - DUPLICATE_WINDOW_SECONDS
    with db() as conn:
        return conn.execute(
            """
            SELECT id, analysis_text
            FROM submissions
            WHERE payload_hash = ? AND created_at >= ? AND analysis_text IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (payload_hash, cutoff),
        ).fetchone()


def client_identity(request: Request, user_id: str | None = None) -> str:
    host = request.client.host if request.client else "unknown"
    return user_id or host


def enforce_rate_limit(key: str, limit: int) -> None:
    now = time.time()
    bucket = _rate_buckets[key]
    while bucket and bucket[0] <= now - RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    bucket.append(now)


def call_external_analysis(payload: dict[str, Any]) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        raise HTTPException(status_code=503, detail="external analysis API is not configured")

    prompt = (
        "请基于以下问卷回答给出简短、谨慎的心理状态分析。"
        "不要做医学诊断，输出应包含压力水平、主要风险和一条建议。\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是谨慎的心理问卷分析助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail="external analysis API failed") from exc


def write_result(submission_id: str, analysis: str) -> None:
    (RESULTS_DIR / f"{submission_id}.json").write_text(
        json.dumps({"submission_id": submission_id, "analysis": analysis}, ensure_ascii=False),
        encoding="utf-8",
    )
