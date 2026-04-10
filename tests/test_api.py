from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app
from services import AssessmentSession


def test_assessment_session_flow() -> None:
    session = AssessmentSession(scoring_model="binary_2pl", max_items=3, device="cpu")

    first = session.next_question()
    assert first is not None
    accepted = session.submit_response(str(first["item_id"]), 5)

    assert accepted["accepted"] is True
    assert accepted["progress"]["answered"] == 1
    assert session.result()["progress"]["answered"] == 1


def test_fastapi_session_flow() -> None:
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    created = client.post(
        "/sessions",
        json={
            "scoring_model": "binary_2pl",
            "max_items": 2,
            "device": "cpu",
            "coverage_min_per_dimension": 1,
        },
    )
    assert created.status_code == 200
    payload = created.json()
    session_id = payload["session_id"]
    question = payload["next_question"]
    assert question["item_id"]

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
    assert "classical_big5" in result_payload
    assert "dimension_answer_counts" in result_payload
