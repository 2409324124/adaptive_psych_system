from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import uuid

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import api.app as api_module
from api.app import app
from engine.database import Base, UserSessionRecord
from services import AssessmentSession, ProgressEstimator, ResultInterpreter, SessionStore


def test_assessment_session_flow() -> None:
    session = AssessmentSession(scoring_model="binary_2pl", max_items=3, min_items=1, stop_mean_standard_error=5.0, device="cpu")

    first = session.next_question()
    assert first is not None
    second_read = session.next_question()
    assert second_read is not None
    assert second_read["item_id"] == first["item_id"]
    accepted = session.submit_response(str(first["item_id"]), 5)

    assert accepted["accepted"] is True
    assert accepted["progress"]["answered"] == 1
    result = session.result()
    assert result["progress"]["answered"] == 1
    assert "interpretation" in result
    assert "standard_errors" in result
    assert "uncertainty" in result
    assert result["param_mode"] == "keyed"
    assert result["key_aligned"] is True
    assert result["progress_estimate"]["estimate_source"] == "lookup_table"


def test_progress_estimator_falls_back_for_unknown_combo() -> None:
    estimate = ProgressEstimator().estimate(
        param_mode="legacy",
        scoring_model="grm",
        coverage_min_per_dimension=9,
        stop_mean_standard_error=0.11,
        answered=0,
        max_items=12,
        complete=False,
        min_items_met=False,
        coverage_ready=False,
        standard_error_ready=False,
        stability_ready=False,
        stopped_by="min_items_gate",
        early_stop_candidate=False,
        confirmation_items_remaining=0,
    )
    assert estimate["estimated_total_items"] == 12
    assert estimate["estimate_source"] == "fallback_max_items"


def test_cached_active_item_refreshes_progress_snapshot() -> None:
    session = AssessmentSession(scoring_model="binary_2pl", max_items=5, min_items=2, device="cpu")
    first = session.next_question()
    assert first is not None
    session.min_items = 1
    again = session.next_question()
    assert again is not None
    assert again["progress"]["min_items"] == 1
    assert again["progress_estimate"]["display_answered"] == 1


def test_session_does_not_stop_before_min_items() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=6,
        min_items=3,
        coverage_min_per_dimension=0,
        stop_mean_standard_error=10.0,
        device="cpu",
    )
    first = session.next_question()
    assert first is not None
    session.submit_response(str(first["item_id"]), 5)
    assert session.is_complete is False


def test_session_does_not_stop_before_checkpoints_even_when_uncertainty_is_low() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=12,
        min_items=1,
        coverage_min_per_dimension=0,
        stop_mean_standard_error=30.0,
        stop_stability_score=0.0,
        device="cpu",
    )
    first = session.next_question()
    assert first is not None
    session.submit_response(str(first["item_id"]), 5)
    assert session.is_complete is False
    assert session.progress()["stopped_by"] in {"screening_gate", "stability_gate"}


def test_session_checkpoint_starts_confirmation_window_instead_of_stopping_immediately() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=5,
        coverage_min_per_dimension=2,
        stop_mean_standard_error=0.65,
        stop_stability_score=0.7,
        device="cpu",
    )
    session.router.answered_indices = set(range(12))
    session.router.dimension_answer_counts = lambda: {dimension: 2 for dimension in session.router.dimensions}
    session.router.uncertainty_summary = lambda: {"mean_standard_error": 0.79, "max_standard_error": 0.9, "confidence_ready": False}
    session.stability = lambda: {"stability_score": 0.88, "stability_ready": True, "stability_stage": "stable"}

    session._advance_candidate_state()
    progress = session.progress()

    assert progress["complete"] is False
    assert progress["stopped_by"] == "confirmation_window"
    assert progress["early_stop_candidate"] is True
    assert progress["candidate_checkpoint"] == 12
    assert progress["confirmation_items_remaining"] == 2
    assert progress["precision_mode"] == "confirmation"
    assert progress["effective_stop_mean_standard_error"] == progress["screening_stop_mean_standard_error"]


def test_session_checkpoint_can_start_candidate_before_stability_is_ready() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=5,
        coverage_min_per_dimension=2,
        stop_mean_standard_error=0.65,
        stop_stability_score=0.7,
        device="cpu",
    )
    session.router.answered_indices = set(range(15))
    session.router.dimension_answer_counts = lambda: {dimension: 3 for dimension in session.router.dimensions}
    session.router.uncertainty_summary = lambda: {"mean_standard_error": 0.89, "max_standard_error": 0.95, "confidence_ready": False}
    session.stability = lambda: {"stability_score": 0.64, "stability_ready": False, "stability_stage": "mixed"}

    session._advance_candidate_state()
    progress = session.progress()

    assert progress["complete"] is False
    assert progress["stopped_by"] == "confirmation_window"
    assert progress["early_stop_candidate"] is True
    assert progress["candidate_checkpoint"] == 15
    assert progress["confirmation_items_remaining"] == 2
    assert progress["stability_ready"] is False


def test_session_can_start_missed_checkpoint_candidate_after_crossing_threshold() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=5,
        coverage_min_per_dimension=2,
        stop_mean_standard_error=0.65,
        stop_stability_score=0.7,
        device="cpu",
    )
    session.router.answered_indices = set(range(13))
    session.router.dimension_answer_counts = lambda: {dimension: 3 for dimension in session.router.dimensions}
    session.router.uncertainty_summary = lambda: {"mean_standard_error": 0.94, "max_standard_error": 1.0, "confidence_ready": False}
    session.stability = lambda: {"stability_score": 0.82, "stability_ready": True, "stability_stage": "stable"}

    session._advance_candidate_state()
    progress = session.progress()

    assert progress["complete"] is False
    assert progress["stopped_by"] == "confirmation_window"
    assert progress["early_stop_candidate"] is True
    assert progress["candidate_checkpoint"] == 12
    assert progress["confirmation_items_remaining"] == 2


def test_session_confirms_candidate_after_two_more_items() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=5,
        coverage_min_per_dimension=2,
        stop_mean_standard_error=0.65,
        stop_stability_score=0.7,
        device="cpu",
    )
    session.router.answered_indices = set(range(12))
    session.router.dimension_answer_counts = lambda: {dimension: 2 for dimension in session.router.dimensions}
    session.router.uncertainty_summary = lambda: {"mean_standard_error": 0.79, "max_standard_error": 0.9, "confidence_ready": False}
    session.stability = lambda: {"stability_score": 0.88, "stability_ready": True, "stability_stage": "stable"}
    session._advance_candidate_state()
    session.confirmation_items_remaining = 0
    session.confirmation_result = "confirmed"
    session.candidate_snapshot = {
        "checkpoint": 12,
        "target_mean_standard_error": 0.80,
        "mean_standard_error": 0.79,
        "stability_score": 0.88,
        "top_trait": session.router.dimensions[-1],
        "lowest_trait": session.router.dimensions[0],
    }
    session._trait_edges = lambda: (session.router.dimensions[-1], session.router.dimensions[0])

    progress = session.progress()

    assert progress["complete"] is True
    assert progress["stopped_by"] == "screening_confirmed"
    assert progress["precision_mode"] == "confirmation"


def test_session_confirmation_can_succeed_without_stability_ready() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=5,
        coverage_min_per_dimension=2,
        stop_mean_standard_error=0.65,
        stop_stability_score=0.7,
        device="cpu",
    )
    session.router.answered_indices = set(range(12))
    session.router.dimension_answer_counts = lambda: {dimension: 3 for dimension in session.router.dimensions}
    session.candidate_snapshot = {
        "checkpoint": 12,
        "target_mean_standard_error": 0.80,
        "mean_standard_error": 0.79,
        "stability_score": 0.55,
        "top_trait": session.router.dimensions[-1],
        "lowest_trait": session.router.dimensions[0],
    }
    session.confirmation_result = "confirmed"
    session._trait_edges = lambda: (session.router.dimensions[-1], session.router.dimensions[0])
    session.router.uncertainty_summary = lambda: {"mean_standard_error": 0.81, "max_standard_error": 0.9, "confidence_ready": False}
    session.stability = lambda: {"stability_score": 0.55, "stability_ready": False, "stability_stage": "mixed"}

    progress = session.progress()

    assert progress["complete"] is True
    assert progress["stopped_by"] == "screening_confirmed"
    assert progress["precision_mode"] == "confirmation"
    assert progress["stability_ready"] is False


def test_session_clears_candidate_when_confirmation_fails() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=5,
        coverage_min_per_dimension=2,
        stop_mean_standard_error=0.65,
        stop_stability_score=0.7,
        device="cpu",
    )
    session.router.answered_indices = set(range(12))
    session.router.dimension_answer_counts = lambda: {dimension: 2 for dimension in session.router.dimensions}
    uncertainty_calls = {"count": 0}

    def uncertainty_summary():
        uncertainty_calls["count"] += 1
        if uncertainty_calls["count"] == 1:
            return {"mean_standard_error": 0.79, "max_standard_error": 0.9, "confidence_ready": False}
        return {"mean_standard_error": 1.05, "max_standard_error": 1.1, "confidence_ready": False}

    session.router.uncertainty_summary = uncertainty_summary
    session.stability = lambda: {"stability_score": 0.88, "stability_ready": True, "stability_stage": "stable"}
    session._advance_candidate_state()
    session.confirmation_items_remaining = 1
    session._advance_candidate_state()

    progress = session.progress()

    assert progress["complete"] is False
    assert progress["early_stop_candidate"] is False
    assert progress["candidate_checkpoint"] is None
    assert progress["stopped_by"] == "screening_gate"


def test_session_clears_invalid_confirmed_state_and_advances_to_next_checkpoint() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=5,
        coverage_min_per_dimension=2,
        stop_mean_standard_error=0.65,
        stop_stability_score=0.7,
        device="cpu",
    )
    session.router.answered_indices = set(range(16))
    session.router.dimension_answer_counts = lambda: {dimension: 3 for dimension in session.router.dimensions}
    session.router.uncertainty_summary = lambda: {"mean_standard_error": 0.89, "max_standard_error": 0.95, "confidence_ready": False}
    session.stability = lambda: {"stability_score": 0.6, "stability_ready": False, "stability_stage": "mixed"}
    session.candidate_snapshot = {
        "checkpoint": 12,
        "target_mean_standard_error": 0.80,
        "mean_standard_error": 0.79,
        "stability_score": 0.88,
        "top_trait": session.router.dimensions[-1],
        "lowest_trait": session.router.dimensions[0],
    }
    session.confirmation_result = "confirmed"
    session.candidate_checkpoint = 12
    session.attempted_candidate_checkpoints = [12]
    session._trait_edges = lambda: (session.router.dimensions[0], session.router.dimensions[-1])

    progress = session.progress()

    assert progress["complete"] is False
    assert progress["candidate_checkpoint"] == 12
    assert progress["stopped_by"] == "confirmation_window"
    assert progress["precision_mode"] == "confirmation"
    assert session.confirmation_result == "confirmed"
    assert session.attempted_candidate_checkpoints == [12]

    session._advance_candidate_state()

    assert session.confirmation_result is None
    assert session.candidate_checkpoint == 15
    assert session.early_stop_candidate is True
    assert session.confirmation_items_remaining == 2


def test_session_confirmation_rejects_low_stability_score_even_when_se_and_traits_hold() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=5,
        coverage_min_per_dimension=2,
        stop_mean_standard_error=0.65,
        stop_stability_score=0.7,
        device="cpu",
    )
    session.router.answered_indices = set(range(12))
    session.router.dimension_answer_counts = lambda: {dimension: 3 for dimension in session.router.dimensions}
    session.candidate_snapshot = {
        "checkpoint": 12,
        "target_mean_standard_error": 0.80,
        "mean_standard_error": 0.79,
        "stability_score": 0.45,
        "top_trait": session.router.dimensions[-1],
        "lowest_trait": session.router.dimensions[0],
    }
    session.confirmation_result = "confirmed"
    session.attempted_candidate_checkpoints = [12]
    session._trait_edges = lambda: (session.router.dimensions[-1], session.router.dimensions[0])
    session.router.uncertainty_summary = lambda: {"mean_standard_error": 0.81, "max_standard_error": 0.9, "confidence_ready": False}
    session.stability = lambda: {"stability_score": 0.45, "stability_ready": False, "stability_stage": "mixed"}

    progress = session.progress()

    assert progress["complete"] is False
    assert progress["stopped_by"] == "stability_gate"
    assert progress["precision_mode"] == "screening"
    assert session.confirmation_result == "confirmed"


def test_session_wraps_when_screening_plateau_persists_past_final_checkpoint() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=5,
        coverage_min_per_dimension=2,
        stop_mean_standard_error=0.65,
        stop_stability_score=0.7,
        device="cpu",
    )
    session.router.answered_indices = set(range(19))
    session.router.dimension_answer_counts = lambda: {dimension: 3 for dimension in session.router.dimensions}
    session.router.uncertainty_summary = lambda: {"mean_standard_error": 0.9, "max_standard_error": 1.0, "confidence_ready": False}
    session.stability = lambda: {"stability_score": 0.92, "stability_ready": True, "stability_stage": "stable"}

    progress = session.progress()

    assert progress["complete"] is True
    assert progress["stopped_by"] == "screening_plateau"
    assert progress["precision_mode"] == "screening_plateau"
    assert progress["screening_threshold_ready"] is False


def test_session_sixteen_item_screening_pass_uses_latest_eligible_checkpoint() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=5,
        coverage_min_per_dimension=2,
        stop_mean_standard_error=0.65,
        stop_stability_score=0.7,
        device="cpu",
    )
    session.router.answered_indices = set(range(16))
    session.router.dimension_answer_counts = lambda: {dimension: 3 for dimension in session.router.dimensions}
    session.router.uncertainty_summary = lambda: {"mean_standard_error": 0.84, "max_standard_error": 0.9, "confidence_ready": False}
    session.stability = lambda: {"stability_score": 0.8, "stability_ready": True, "stability_stage": "stable"}

    session._advance_candidate_state()
    progress = session.progress()

    assert progress["complete"] is False
    assert progress["precision_mode"] == "confirmation"
    assert progress["stopped_by"] == "confirmation_window"
    assert progress["early_stop_candidate"] is True
    assert progress["candidate_checkpoint"] == 15


def test_session_progress_reports_candidate_ready_as_confirmation_window() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=5,
        coverage_min_per_dimension=2,
        stop_mean_standard_error=0.65,
        stop_stability_score=0.7,
        device="cpu",
    )
    session.router.answered_indices = set(range(12))
    session.router.dimension_answer_counts = lambda: {dimension: 2 for dimension in session.router.dimensions}
    session.router.uncertainty_summary = lambda: {"mean_standard_error": 0.79, "max_standard_error": 0.9, "confidence_ready": False}
    session.stability = lambda: {"stability_score": 0.88, "stability_ready": True, "stability_stage": "stable"}

    progress = session.progress()

    assert progress["complete"] is False
    assert progress["stopped_by"] == "confirmation_window"
    assert progress["precision_mode"] == "confirmation"


def test_session_waits_for_stability_before_stopping() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=8,
        min_items=1,
        coverage_min_per_dimension=0,
        stop_mean_standard_error=30.0,
        stop_stability_score=0.95,
        device="cpu",
    )
    first = session.next_question()
    assert first is not None
    session.submit_response(str(first["item_id"]), 3)
    assert session.is_complete is False
    assert session.progress()["stopped_by"] == "stability_gate"


def test_session_progress_marks_neutral_only_pattern_as_unstable() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=8,
        min_items=1,
        coverage_min_per_dimension=0,
        stop_mean_standard_error=30.0,
        stop_stability_score=0.7,
        device="cpu",
    )
    for _ in range(3):
        item = session.next_question()
        assert item is not None
        session.submit_response(str(item["item_id"]), 3)

    progress = session.progress()
    assert progress["stability_score"] <= 0.45
    assert progress["stability_ready"] is False
    assert progress["stability_stage"] != "stable"


def test_session_progress_marks_all_extreme_pattern_as_unstable() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=8,
        min_items=1,
        coverage_min_per_dimension=0,
        stop_mean_standard_error=30.0,
        stop_stability_score=0.7,
        device="cpu",
    )
    for _ in range(3):
        item = session.next_question()
        assert item is not None
        session.submit_response(str(item["item_id"]), 5)

    progress = session.progress()
    assert progress["stability_score"] <= 0.55
    assert progress["stability_ready"] is False
    assert progress["stability_stage"] != "stable"


def test_session_does_not_early_stop_after_sixteen_extreme_answers() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=1,
        coverage_min_per_dimension=0,
        stop_mean_standard_error=30.0,
        stop_stability_score=0.7,
        device="cpu",
    )

    for _ in range(16):
        item = session.next_question()
        assert item is not None
        session.submit_response(str(item["item_id"]), 5)

    progress = session.progress()
    assert session.is_complete is False
    assert progress["answered"] == 16
    assert progress["stopped_by"] == "stability_gate"
    assert progress["stability_ready"] is False
    assert progress["stability_score"] <= 0.35


def test_result_interpreter_marks_low_evidence_traits() -> None:
    interpreter = ResultInterpreter(low_evidence_threshold=2)
    payload = interpreter.interpret(
        irt_t_scores={
            "extraversion": 58.0,
            "agreeableness": 51.0,
            "conscientiousness": 43.5,
            "emotional_stability": 49.5,
            "intellect": 56.0,
        },
        dimension_answer_counts={
            "extraversion": 2,
            "agreeableness": 2,
            "conscientiousness": 1,
            "emotional_stability": 3,
            "intellect": 1,
        },
    )
    assert "这一轮最突出的信号" in payload["overview"]
    assert payload["low_evidence_traits"] == ["尽责", "智力/开放"]
    assert any("偏低一些" in line for line in payload["lowlights"])
    assert payload["structured_summary"]["headline_trait"]["label"] == "外向"


def test_session_store_json_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions")
    session = store.create_session(
        scoring_model="binary_2pl",
        max_items=1,
        min_items=1,
        device="cpu",
        param_mode="keyed",
        param_path=None,
        coverage_min_per_dimension=1,
        stop_mean_standard_error=5.0,
        stop_stability_score=0.0,
    )
    first = session.next_question()
    assert first is not None
    session.submit_response(str(first["item_id"]), 4)
    session.add_comment("我觉得自己平时又拧巴又上头。")
    store.save_session(session)

    reloaded = SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions").get_session(session.session_id)
    assert reloaded.progress()["answered"] == 1
    assert reloaded.path[0]["item_id"] == first["item_id"]
    assert reloaded.path[0]["theta_after"] == reloaded.router.history[0]["theta_after"]
    assert reloaded.path[0]["text_zh"]
    assert reloaded.param_mode == "keyed"
    assert reloaded.router.key_aligned is True
    assert reloaded.comment_submitted is True
    assert reloaded.user_comments == ["我觉得自己平时又拧巴又上头。"]


def test_session_comments_require_completion() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=3,
        min_items=2,
        coverage_min_per_dimension=1,
        stop_mean_standard_error=0.85,
        stop_stability_score=0.7,
        device="cpu",
    )

    try:
        session.add_comment("还没做完，我先插一句。")
    except ValueError as exc:
        assert "complete" in str(exc)
    else:
        raise AssertionError("Comments should be rejected before completion.")


def test_snapshot_preserves_active_item(tmp_path: Path) -> None:
    store = SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions")
    session = store.create_session(
        scoring_model="binary_2pl",
        max_items=5,
        min_items=2,
        device="cpu",
        param_mode=None,
        param_path=None,
        coverage_min_per_dimension=1,
        stop_mean_standard_error=0.85,
        stop_stability_score=0.7,
    )
    first = session.next_question()
    assert first is not None
    store.save_session(session)

    reloaded = SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions").get_session(session.session_id)
    assert reloaded.active_item is not None
    assert reloaded.active_item["text_zh"]
    assert reloaded.next_question()["item_id"] == first["item_id"]


def test_session_store_expires_idle_sessions(tmp_path: Path) -> None:
    store = SessionStore(backend="json", ttl_seconds=0, storage_dir=tmp_path / "sessions")
    session = store.create_session(
        scoring_model="binary_2pl",
        max_items=3,
        min_items=2,
        device="cpu",
        param_mode=None,
        param_path=None,
        coverage_min_per_dimension=1,
        stop_mean_standard_error=0.85,
        stop_stability_score=0.7,
    )
    session.updated_at = "2000-01-01T00:00:00+00:00"
    store.sessions[session.session_id] = session
    (tmp_path / "sessions" / f"{session.session_id}.json").write_text(
        json.dumps(session.snapshot(), ensure_ascii=False),
        encoding="utf-8",
    )
    store.cleanup_expired()
    try:
        store.get_session(session.session_id)
    except KeyError:
        assert True
    else:
        raise AssertionError("Expired session should not be available.")


def test_fastapi_session_flow(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "cat_psych.db"
    test_engine = create_engine(f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr(
        api_module,
        "SESSION_STORE",
        SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions"),
    )
    monkeypatch.setattr(api_module, "SessionLocal", test_session_local)
    monkeypatch.setattr(
        api_module,
        "analyze_personality",
        lambda ocean_scores, user_comments, structured_summary=None: {
            "category_key": "Siberian Black",
            "analysis": f"Mock analysis for {len(user_comments)} comments.",
        },
    )
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    created = client.post(
        "/sessions",
        json={
            "scoring_model": "binary_2pl",
            "max_items": 2,
            "min_items": 1,
            "device": "cpu",
            "coverage_min_per_dimension": 1,
            "stop_mean_standard_error": 5.0,
            "stop_stability_score": 0.0,
        },
    )
    assert created.status_code == 200
    payload = created.json()
    session_id = payload["session_id"]
    assert str(uuid.UUID(session_id)) == session_id
    question = payload["next_question"]
    assert question["item_id"]
    assert question["text_zh"]
    assert payload["min_items"] == 1
    assert payload["param_mode"] == "keyed"
    assert payload["key_aligned"] is True
    assert payload["progress_estimate"]["estimated_total_items"] == 2
    assert payload["progress_estimate"]["estimate_source"] == "fallback_max_items"
    assert payload["stop_stability_score"] == 0.0
    summary = client.get(f"/sessions/{session_id}")
    assert summary.status_code == 200
    assert summary.json()["session_id"] == session_id
    assert summary.json()["min_items"] == 1
    assert summary.json()["progress"]["stopped_by"] == "min_items_gate"
    assert "stability_score" in summary.json()["progress"]

    first_response = client.post(
        f"/sessions/{session_id}/responses",
        json={"item_id": question["item_id"], "response": 5},
    )
    assert first_response.status_code == 200
    next_question = first_response.json()["next_question"]
    assert next_question is not None
    assert next_question["text_zh"]

    second_response = client.post(
        f"/sessions/{session_id}/responses",
        json={"item_id": next_question["item_id"], "response": 4},
    )
    assert second_response.status_code == 200
    assert second_response.json()["complete"] is True

    comment_response = client.post(
        f"/sessions/{session_id}/comments",
        json={"comment": "我平时确实是表面冷静，内心在疯狂写分支逻辑。"},
    )
    assert comment_response.status_code == 200
    assert comment_response.json()["comment_submitted"] is True

    result = client.get(f"/sessions/{session_id}/result")
    assert result.status_code == 200
    result_payload = result.json()
    assert result_payload["progress"]["complete"] is True
    assert result_payload["runtime_state"] == "missing"
    assert result_payload["persistence_state"] == "persisted"
    assert "irt_t_scores" in result_payload
    assert "standard_errors" in result_payload
    assert "uncertainty" in result_payload
    assert "classical_big5" in result_payload
    assert "dimension_answer_counts" in result_payload
    assert "interpretation" in result_payload
    assert "overview" in result_payload["interpretation"]
    assert "param_metadata" in result_payload
    assert "stopped_by" in result_payload["progress"]
    assert "progress_estimate" in result_payload
    assert "stability" in result_payload
    assert "stability_stage" in result_payload["progress"]
    assert result_payload["cat_category"] == "Siberian Black"
    assert result_payload["cat_name"] == "【废土独狼】西伯利亚黑猫"
    assert result_payload["cat_image"] == "/static/cats/siberian_black.png"
    assert result_payload["cat_image_position"] == "40% 34%"
    assert "Mock analysis" in result_payload["cat_analysis"]
    assert result_payload["path"][0]["text_zh"]

    finalized_summary = client.get(f"/sessions/{session_id}")
    assert finalized_summary.status_code == 404

    finalized_next = client.get(f"/sessions/{session_id}/next")
    assert finalized_next.status_code == 404

    repeated = client.get(f"/sessions/{session_id}/result")
    assert repeated.status_code == 200
    assert repeated.json()["cat_analysis"] == result_payload["cat_analysis"]

    exported = client.get(f"/sessions/{session_id}/export")
    assert exported.status_code == 200
    assert "attachment; filename=" in exported.headers["content-disposition"]
    assert exported.json()["session_id"] == session_id
    assert exported.json()["param_mode"] == "keyed"
    assert exported.json()["progress_estimate"]["estimate_source"] == "fallback_max_items"
    assert "stability" in exported.json()
    assert exported.json()["cat_category"] == "Siberian Black"
    assert exported.json()["path"][0]["text_zh"]

    deleted = client.delete(f"/sessions/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    replayed = client.get(f"/sessions/{session_id}/result")
    assert replayed.status_code == 200
    assert replayed.json()["cat_analysis"] == result_payload["cat_analysis"]
    assert replayed.json()["runtime_state"] == "missing"

    shared = client.get(f"/results/{session_id}")
    assert shared.status_code == 200
    assert shared.json()["cat_name"] == "【废土独狼】西伯利亚黑猫"
    assert shared.json()["session_id"] == session_id

    restarted = client.post(f"/sessions/{session_id}/restart")
    assert restarted.status_code == 409

    with test_session_local() as db:
        persisted = db.get(UserSessionRecord, session_id)
        assert persisted is not None
        assert persisted.cat_category == "Siberian Black"


def test_finalized_session_blocks_runtime_mutations_but_keeps_persisted_result(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "cat_psych.db"
    test_engine = create_engine(f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr(
        api_module,
        "SESSION_STORE",
        SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions"),
    )
    monkeypatch.setattr(api_module, "SessionLocal", test_session_local)
    monkeypatch.setattr(
        api_module,
        "analyze_personality",
        lambda ocean_scores, user_comments, structured_summary=None: {
            "category_key": "Scottish Fold",
            "analysis": "Finalized lifecycle test analysis.",
        },
    )
    client = TestClient(app)

    created = client.post(
        "/sessions",
        json={
            "scoring_model": "binary_2pl",
            "max_items": 2,
            "min_items": 1,
            "device": "cpu",
            "coverage_min_per_dimension": 1,
            "stop_mean_standard_error": 5.0,
            "stop_stability_score": 0.0,
        },
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]
    first_item = created.json()["next_question"]

    first_response = client.post(
        f"/sessions/{session_id}/responses",
        json={"item_id": first_item["item_id"], "response": 4},
    )
    assert first_response.status_code == 200
    second_item = first_response.json()["next_question"]
    assert second_item is not None

    second_response = client.post(
        f"/sessions/{session_id}/responses",
        json={"item_id": second_item["item_id"], "response": 5},
    )
    assert second_response.status_code == 200
    assert second_response.json()["complete"] is True

    finalized = client.get(f"/sessions/{session_id}/result")
    assert finalized.status_code == 200
    assert finalized.json()["persistence_state"] == "persisted"

    rejected_response = client.post(
        f"/sessions/{session_id}/responses",
        json={"item_id": second_item["item_id"], "response": 3},
    )
    assert rejected_response.status_code == 404
    assert rejected_response.json()["detail"] == "Unknown session_id"

    rejected_comment = client.post(
        f"/sessions/{session_id}/comments",
        json={"comment": "This should be rejected after finalization."},
    )
    assert rejected_comment.status_code == 409
    assert "finalized" in rejected_comment.json()["detail"].lower()

    rejected_restart = client.post(f"/sessions/{session_id}/restart")
    assert rejected_restart.status_code == 409
    assert "cannot be restarted" in rejected_restart.json()["detail"].lower()

    shared = client.get(f"/results/{session_id}")
    assert shared.status_code == 200
    assert shared.json()["cat_category"] == "Scottish Fold"


def test_result_fallback_persists_once_and_shared_link_survives(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "cat_psych.db"
    test_engine = create_engine(f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr(
        api_module,
        "SESSION_STORE",
        SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions"),
    )
    monkeypatch.setattr(api_module, "SessionLocal", test_session_local)

    fallback_calls = {"count": 0}

    def fake_analysis(ocean_scores, user_comments, structured_summary=None):
        fallback_calls["count"] += 1
        return {
            "category_key": "Scottish Fold",
            "analysis": f"Fallback-safe analysis with {len(user_comments)} comment(s).",
        }

    monkeypatch.setattr(api_module, "analyze_personality", fake_analysis)
    client = TestClient(app)

    created = client.post(
        "/sessions",
        json={
            "scoring_model": "binary_2pl",
            "max_items": 2,
            "min_items": 1,
            "device": "cpu",
            "coverage_min_per_dimension": 1,
            "stop_mean_standard_error": 5.0,
            "stop_stability_score": 0.0,
        },
    )
    assert created.status_code == 200
    payload = created.json()
    session_id = payload["session_id"]
    first_item = payload["next_question"]

    first_response = client.post(
        f"/sessions/{session_id}/responses",
        json={"item_id": first_item["item_id"], "response": 4},
    )
    assert first_response.status_code == 200
    second_item = first_response.json()["next_question"]
    assert second_item is not None

    second_response = client.post(
        f"/sessions/{session_id}/responses",
        json={"item_id": second_item["item_id"], "response": 5},
    )
    assert second_response.status_code == 200
    assert second_response.json()["complete"] is True

    comment_response = client.post(
        f"/sessions/{session_id}/comments",
        json={"comment": "这轮我先留一句吐槽，看看 fallback 会不会稳定落库。"},
    )
    assert comment_response.status_code == 200

    first_result = client.get(f"/sessions/{session_id}/result")
    assert first_result.status_code == 200
    first_payload = first_result.json()
    assert first_payload["cat_category"] == "Scottish Fold"
    assert first_payload["cat_image_position"] == "50% 30%"
    assert "Fallback-safe analysis" in first_payload["cat_analysis"]

    second_result = client.get(f"/sessions/{session_id}/result")
    assert second_result.status_code == 200
    second_payload = second_result.json()
    assert second_payload["cat_analysis"] == first_payload["cat_analysis"]
    assert second_payload["cat_category"] == first_payload["cat_category"]
    assert fallback_calls["count"] == 1

    with test_session_local() as db:
        persisted = db.get(UserSessionRecord, session_id)
        assert persisted is not None
        assert persisted.cat_category == "Scottish Fold"
        assert persisted.raw_responses["user_comments"] == ["这轮我先留一句吐槽，看看 fallback 会不会稳定落库。"]

    deleted = client.delete(f"/sessions/{session_id}")
    assert deleted.status_code == 200

    shared = client.get(f"/results/{session_id}")
    assert shared.status_code == 200
    assert shared.json()["cat_category"] == "Scottish Fold"
    assert shared.json()["cat_analysis"] == first_payload["cat_analysis"]


def test_api_rejects_comment_before_completion(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "cat_psych.db"
    test_engine = create_engine(f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr(
        api_module,
        "SESSION_STORE",
        SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions"),
    )
    monkeypatch.setattr(api_module, "SessionLocal", test_session_local)
    client = TestClient(app)

    created = client.post(
        "/sessions",
        json={
            "scoring_model": "binary_2pl",
            "max_items": 3,
            "min_items": 2,
            "device": "cpu",
            "coverage_min_per_dimension": 1,
            "stop_mean_standard_error": 0.85,
            "stop_stability_score": 0.7,
        },
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    comment_response = client.post(
        f"/sessions/{session_id}/comments",
        json={"comment": "我还没做完，但想先吐槽。"},
    )
    assert comment_response.status_code == 409
    assert "complete" in comment_response.json()["detail"]


def test_concurrent_result_requests_only_invoke_analysis_once(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "cat_psych.db"
    test_engine = create_engine(f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr(
        api_module,
        "SESSION_STORE",
        SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions"),
    )
    monkeypatch.setattr(api_module, "SessionLocal", test_session_local)

    analysis_calls = {"count": 0}

    def fake_analysis(ocean_scores, user_comments, structured_summary=None):
        analysis_calls["count"] += 1
        time.sleep(0.2)
        return {
            "category_key": "Scottish Fold",
            "analysis": "Concurrent-safe analysis.",
        }

    monkeypatch.setattr(api_module, "analyze_personality", fake_analysis)

    with TestClient(app) as client:
        created = client.post(
            "/sessions",
            json={
                "scoring_model": "binary_2pl",
                "max_items": 1,
                "min_items": 1,
                "device": "cpu",
                "coverage_min_per_dimension": 1,
                "stop_mean_standard_error": 5.0,
                "stop_stability_score": 0.0,
            },
        )
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        question = created.json()["next_question"]

        answered = client.post(
            f"/sessions/{session_id}/responses",
            json={"item_id": question["item_id"], "response": 5},
        )
        assert answered.status_code == 200
        assert answered.json()["complete"] is True

    barrier = threading.Barrier(3)
    statuses: list[int] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            with TestClient(app) as worker_client:
                barrier.wait()
                response = worker_client.get(f"/sessions/{session_id}/result")
                statuses.append(response.status_code)
        except Exception as exc:  # pragma: no cover - test failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert not errors
    assert statuses == [200, 200]
    assert analysis_calls["count"] == 1
    assert api_module._PERSIST_LOCKS == {}


def test_get_result_payload_rechecks_db_after_runtime_session_is_retired(monkeypatch) -> None:
    persisted_record = object()
    lookup_calls = {"count": 0}

    def fake_get_persisted_record(session_id, db):
        lookup_calls["count"] += 1
        if lookup_calls["count"] == 1:
            return None
        return persisted_record

    monkeypatch.setattr(api_module, "get_persisted_record", fake_get_persisted_record)
    monkeypatch.setattr(
        api_module,
        "get_session",
        lambda session_id: (_ for _ in ()).throw(HTTPException(status_code=404, detail="Unknown session_id")),
    )
    monkeypatch.setattr(
        api_module,
        "combine_db_record",
        lambda record: {"session_id": "race-session", "cat_category": "Scottish Fold"} if record is persisted_record else {},
    )

    payload = api_module.get_result_payload("race-session", db=object())

    assert payload["session_id"] == "race-session"
    assert payload["cat_category"] == "Scottish Fold"
    assert lookup_calls["count"] == 2


def test_result_persistence_survives_runtime_cleanup_failure(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "cat_psych.db"
    test_engine = create_engine(f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    monkeypatch.setattr(
        api_module,
        "SESSION_STORE",
        SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions"),
    )
    monkeypatch.setattr(api_module, "SessionLocal", test_session_local)
    monkeypatch.setattr(
        api_module,
        "analyze_personality",
        lambda ocean_scores, user_comments, structured_summary=None: {
            "category_key": "Scottish Fold",
            "analysis": "Cleanup-safe analysis.",
        },
    )
    monkeypatch.setattr(api_module.SESSION_STORE, "delete_session", lambda session_id: (_ for _ in ()).throw(RuntimeError("cleanup failed")))

    with TestClient(app) as client:
        created = client.post(
            "/sessions",
            json={
                "scoring_model": "binary_2pl",
                "max_items": 1,
                "min_items": 1,
                "device": "cpu",
                "coverage_min_per_dimension": 1,
                "stop_mean_standard_error": 5.0,
                "stop_stability_score": 0.0,
            },
        )
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        question = created.json()["next_question"]

        answered = client.post(
            f"/sessions/{session_id}/responses",
            json={"item_id": question["item_id"], "response": 5},
        )
        assert answered.status_code == 200
        assert answered.json()["complete"] is True

        result = client.get(f"/sessions/{session_id}/result")
        assert result.status_code == 200
        assert result.json()["cat_category"] == "Scottish Fold"

    with test_session_local() as db:
        persisted = db.get(UserSessionRecord, session_id)
        assert persisted is not None
        assert persisted.cat_category == "Scottish Fold"
    assert api_module._PERSIST_LOCKS == {}
