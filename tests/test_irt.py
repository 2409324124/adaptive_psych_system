from __future__ import annotations

import torch

from engine.classical_scoring import ClassicalBigFiveScorer
from engine.irt_model import AdaptiveMMPIRouter
from engine.math_utils import (
    grm_category_probabilities,
    grm_thresholds_from_location,
    likert_to_binary,
    mirt_2pl_probability,
)
from scripts.simulate_adaptive_sessions import PERSONAS, run_matrix
from scripts.run_cli_assessment import parse_demo_responses, run_assessment


def test_likert_to_binary_rules() -> None:
    assert likert_to_binary(1) == 0.0
    assert likert_to_binary(2) == 0.0
    assert likert_to_binary(3) is None
    assert likert_to_binary(4) == 1.0
    assert likert_to_binary(5) == 1.0
    assert likert_to_binary(3, neutral_policy="zero") == 0.5


def test_classical_big_five_reverse_key_scoring() -> None:
    scorer = ClassicalBigFiveScorer()

    assert scorer.keyed_score(5, 1) == 5
    assert scorer.keyed_score(5, -1) == 1
    assert scorer.centered_score(5.0) == 1.0
    assert scorer.tendency_t_score(1.0) == 60.0


def test_router_loads_item_and_parameter_shapes() -> None:
    router = AdaptiveMMPIRouter(device="cpu")

    assert len(router.items) == 50
    assert router.a.shape == (50, 5)
    assert router.b.shape == (50,)
    assert router.theta.shape == (5,)
    assert router.device.type == "cpu"


def test_2pl_probability_range() -> None:
    router = AdaptiveMMPIRouter(device="cpu")

    probabilities = mirt_2pl_probability(router.theta, router.a, router.b)

    assert probabilities.shape == (50,)
    assert torch.all(probabilities > 0.0)
    assert torch.all(probabilities < 1.0)


def test_binary_router_selects_without_repeating() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl")

    first = router.select_next_item()
    assert first is not None
    router.answer_item(str(first["id"]), 5)

    second = router.select_next_item()
    assert second is not None
    assert second["id"] != first["id"]
    assert router.answered_count == 1
    assert router.remaining_count == 49


def test_binary_router_updates_theta_for_non_neutral_response() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl")
    item = router.select_next_item()
    assert item is not None

    before = router.theta.clone()
    router.answer_item(str(item["id"]), 5)

    assert not torch.allclose(before, router.theta)


def test_binary_router_skips_theta_update_for_neutral_response() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl")
    item = router.select_next_item()
    assert item is not None

    before = router.theta.clone()
    router.answer_item(str(item["id"]), 3)

    assert torch.allclose(before, router.theta)
    assert router.answered_count == 1


def test_grm_probabilities_are_valid_category_distribution() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="grm")
    thresholds = grm_thresholds_from_location(router.b)

    probabilities = grm_category_probabilities(router.theta, router.a, thresholds)

    assert probabilities.shape == (50, 5)
    assert torch.all(probabilities > 0.0)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(50), atol=1e-5)


def test_grm_router_updates_theta_for_likert_response() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="grm")
    item = router.select_next_item()
    assert item is not None

    before = router.theta.clone()
    router.answer_item(str(item["id"]), 5)

    assert not torch.allclose(before, router.theta)


def test_router_can_use_cuda_when_available() -> None:
    if not torch.cuda.is_available():
        return

    router = AdaptiveMMPIRouter(device="cuda", scoring_model="binary_2pl")
    item = router.select_next_item()
    assert item is not None
    router.answer_item(str(item["id"]), 4)

    assert router.theta.device.type == "cuda"


def test_simulation_matrix_runs_both_scoring_models() -> None:
    sessions = run_matrix(
        personas=PERSONAS[:1],
        scoring_models=["binary_2pl", "grm"],
        max_items=3,
        device="cpu",
    )

    assert len(sessions) == 2
    assert {session["scoring_model"] for session in sessions} == {"binary_2pl", "grm"}
    assert all(session["answered_count"] == 3 for session in sessions)
    assert all(len(session["path"]) == 3 for session in sessions)
    assert all("classical_big5" in session for session in sessions)


def test_cli_assessment_demo_mode_runs_without_input() -> None:
    result = run_assessment(
        scoring_model="binary_2pl",
        max_items=3,
        device="cpu",
        demo_responses=parse_demo_responses("5,4,3"),
    )

    assert result["answered_count"] == 3
    assert len(result["path"]) == 3
    assert "irt_t_scores" in result
    assert "classical_big5" in result
