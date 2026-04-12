from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import api.app as api_module
from api.app import app
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


def test_session_can_stop_early_when_uncertainty_goal_is_met() -> None:
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
    assert session.is_complete is True


def test_session_can_finish_at_screening_stage_when_stable_and_precise() -> None:
    session = AssessmentSession(
        scoring_model="binary_2pl",
        max_items=30,
        min_items=5,
        coverage_min_per_dimension=2,
        stop_mean_standard_error=0.65,
        stop_stability_score=0.7,
        device="cpu",
    )
    session.router.answered_indices = set(range(10))
    session.router.dimension_answer_counts = lambda: {dimension: 2 for dimension in session.router.dimensions}
    session.router.uncertainty_summary = lambda: {"mean_standard_error": 0.82, "max_standard_error": 0.9, "confidence_ready": False}
    session.stability = lambda: {"stability_score": 0.88, "stability_ready": True, "stability_stage": "stable"}

    progress = session.progress()

    assert progress["complete"] is True
    assert progress["stopped_by"] == "screening_threshold"
    assert progress["precision_mode"] == "screening"
    assert progress["effective_stop_mean_standard_error"] == progress["screening_stop_mean_standard_error"]


def test_session_wraps_when_screening_plateau_persists_past_trigger() -> None:
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
    session.router.uncertainty_summary = lambda: {"mean_standard_error": 0.9, "max_standard_error": 1.0, "confidence_ready": False}
    session.stability = lambda: {"stability_score": 0.92, "stability_ready": True, "stability_stage": "stable"}

    progress = session.progress()

    assert progress["complete"] is True
    assert progress["stopped_by"] == "screening_plateau"
    assert progress["precision_mode"] == "screening_plateau"
    assert progress["screening_threshold_ready"] is False


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
    assert "Current responses lean most strongly" in payload["overview"]
    assert payload["low_evidence_traits"] == ["Conscientiousness", "Intellect / openness"]
    assert any("lower side" in line for line in payload["lowlights"])


def test_session_store_json_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions")
    session = store.create_session(
        scoring_model="binary_2pl",
        max_items=3,
        min_items=2,
        device="cpu",
        param_mode="keyed",
        param_path=None,
        coverage_min_per_dimension=1,
        stop_mean_standard_error=0.85,
        stop_stability_score=0.7,
    )
    first = session.next_question()
    assert first is not None
    session.submit_response(str(first["item_id"]), 4)
    store.save_session(session)

    reloaded = SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions").get_session(session.session_id)
    assert reloaded.progress()["answered"] == 1
    assert reloaded.path[0]["item_id"] == first["item_id"]
    assert reloaded.path[0]["theta_after"] == reloaded.router.history[0]["theta_after"]
    assert reloaded.param_mode == "keyed"
    assert reloaded.router.key_aligned is True


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
    monkeypatch.setattr(
        api_module,
        "SESSION_STORE",
        SessionStore(backend="json", ttl_seconds=3600, storage_dir=tmp_path / "sessions"),
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
    question = payload["next_question"]
    assert question["item_id"]
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

    second_response = client.post(
        f"/sessions/{session_id}/responses",
        json={"item_id": next_question["item_id"], "response": 4},
    )
    assert second_response.status_code == 200
    assert second_response.json()["complete"] is True

    result = client.get(f"/sessions/{session_id}/result")
    assert result.status_code == 200
    result_payload = result.json()
    assert result_payload["progress"]["complete"] is True
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

    exported = client.get(f"/sessions/{session_id}/export")
    assert exported.status_code == 200
    assert "attachment; filename=" in exported.headers["content-disposition"]
    assert exported.json()["session_id"] == session_id
    assert exported.json()["param_mode"] == "keyed"
    assert exported.json()["progress_estimate"]["estimate_source"] == "fallback_max_items"
    assert "stability" in exported.json()

    restarted = client.post(f"/sessions/{session_id}/restart")
    assert restarted.status_code == 200
    assert restarted.json()["progress"]["answered"] == 0
    assert restarted.json()["next_question"]["item_id"]

    deleted = client.delete(f"/sessions/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
