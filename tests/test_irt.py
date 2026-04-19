from __future__ import annotations

from pathlib import Path
import torch

from engine.classical_scoring import ClassicalBigFiveScorer
from engine.irt_model import AdaptiveMMPIRouter
from engine.param_config import resolve_param_source
from engine.math_utils import (
    binary_fisher_information_matrix,
    grm_fisher_information_matrix,
    grm_category_probabilities,
    grm_thresholds_from_location,
    likert_to_binary,
    mirt_2pl_probability,
    response_to_target,
)
from scripts.benchmark_stopping_rules import DEFAULT_CONFIGS, run_benchmark
from scripts.compare_param_modes import build_comparison
from scripts.simulate_adaptive_sessions import PERSONAS, RESPONSE_STYLES, run_matrix, run_session
from scripts.run_cli_assessment import parse_demo_responses, run_assessment
from services.stability_analyzer import StabilityAnalyzer


def test_likert_to_binary_rules() -> None:
    assert likert_to_binary(1) == 0.0
    assert likert_to_binary(2) == 0.0
    assert likert_to_binary(3) is None
    assert likert_to_binary(4) == 1.0
    assert likert_to_binary(5) == 1.0
    assert likert_to_binary(3, neutral_policy="zero") == 0.5


def test_response_to_target_supports_llm_soft_weights() -> None:
    assert response_to_target(1, source="binary") == 1.0
    assert response_to_target(0, source="binary") == 0.0
    assert response_to_target(0.75, source="llm") == 0.75


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


def test_binary_fisher_information_matrix_matches_trait_dimensions() -> None:
    router = AdaptiveMMPIRouter(device="cpu")
    item = router.select_next_item()
    assert item is not None

    matrix = binary_fisher_information_matrix(
        router.theta,
        router.a[int(item["index"])],
        router.b[int(item["index"])],
    )

    assert matrix.shape == (len(router.dimensions), len(router.dimensions))
    assert torch.allclose(matrix, matrix.T, atol=1e-6)


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


def test_router_coverage_constraint_prioritizes_undercovered_dimensions() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl", coverage_min_per_dimension=1)
    seen_dimensions = set()

    for _ in range(len(router.dimensions)):
        item = router.select_next_item()
        assert item is not None
        seen_dimensions.add(str(item["dimension"]))
        router.answer_item(str(item["id"]), 4)

    assert seen_dimensions == set(router.dimensions)


def test_binary_router_updates_theta_for_non_neutral_response() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl")
    item = router.select_next_item()
    assert item is not None

    before = router.theta.clone()
    router.answer_item(str(item["id"]), 5)

    assert not torch.allclose(before, router.theta)


def test_reverse_key_item_pushes_theta_in_opposite_direction() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl")
    reverse_item = next(item for item in router.items if item.key == -1)

    before = router.theta.clone()
    router.answer_item(reverse_item.item_id, 5)
    delta = router.theta - before

    assert torch.dot(delta, router.a[reverse_item.index]).item() < 0.0


def test_key_aligned_params_do_not_double_flip_reverse_items() -> None:
    param_path = Path(__file__).resolve().parents[1] / "data" / "mock_params_keyed.pt"
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl", param_path=param_path)
    reverse_item = next(item for item in router.items if item.key == -1)

    before = router.theta.clone()
    record = router.answer_item(reverse_item.item_id, 5)
    delta = router.theta - before

    assert router.key_aligned is True
    assert record["keyed_response"] == 5
    assert torch.dot(delta, router.a[reverse_item.index]).item() > 0.0


def test_router_exposes_item_level_information_matrix() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl")
    item = router.select_next_item()
    assert item is not None

    matrix = router.fisher_information_matrix(str(item["id"]))
    assert matrix.shape == (len(router.dimensions), len(router.dimensions))
    assert torch.allclose(matrix, matrix.T, atol=1e-6)


def test_grm_router_exposes_item_level_information_matrix() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="grm")
    item = router.select_next_item()
    assert item is not None

    matrix = grm_fisher_information_matrix(
        router.theta,
        router.a[int(item["index"])],
        router.thresholds[int(item["index"])],
    )
    assert matrix.shape == (len(router.dimensions), len(router.dimensions))
    assert torch.allclose(matrix, matrix.T, atol=1e-6)


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


def test_binary_router_accepts_llm_soft_response_update() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl")
    item = router.select_next_item()
    assert item is not None

    before = router.theta.clone()
    record = router.update_theta(str(item["id"]), 0.9, source="llm", response_weight=0.5)

    assert not torch.allclose(before, router.theta)
    assert record["response_source"] == "llm"
    assert record["response_weight"] == 0.5


def test_information_matrix_accumulates_after_response() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl")
    item = router.select_next_item()
    assert item is not None

    before = router.cumulative_information_matrix()
    router.answer_item(str(item["id"]), 5)
    after = router.cumulative_information_matrix()

    assert torch.trace(after) > torch.trace(before)


def test_standard_errors_are_finite_after_updates() -> None:
    router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl", coverage_min_per_dimension=1)
    for _ in range(3):
        item = router.select_next_item()
        assert item is not None
        router.answer_item(str(item["id"]), 4)

    standard_errors = router.standard_errors()
    uncertainty = router.uncertainty_summary()

    assert all(value > 0.0 for value in standard_errors.values())
    assert uncertainty["mean_standard_error"] > 0.0
    assert uncertainty["max_standard_error"] > 0.0


def test_stability_analyzer_rates_stable_sequence_higher_than_wavering() -> None:
    analyzer = StabilityAnalyzer()
    stable_path = [
        {"dimension": "extraversion", "response": 5, "keyed_response": 5},
        {"dimension": "extraversion", "response": 4, "keyed_response": 4},
        {"dimension": "agreeableness", "response": 5, "keyed_response": 5},
        {"dimension": "agreeableness", "response": 4, "keyed_response": 4},
    ]
    stable_history = [
        {"theta_before": [0.0, 0.0], "theta_after": [0.12, 0.05]},
        {"theta_before": [0.12, 0.05], "theta_after": [0.18, 0.08]},
        {"theta_before": [0.18, 0.08], "theta_after": [0.22, 0.1]},
        {"theta_before": [0.22, 0.1], "theta_after": [0.25, 0.12]},
    ]
    wavering_path = [
        {"dimension": "extraversion", "response": 5, "keyed_response": 5},
        {"dimension": "extraversion", "response": 1, "keyed_response": 1},
        {"dimension": "agreeableness", "response": 3, "keyed_response": 3},
        {"dimension": "agreeableness", "response": 5, "keyed_response": 5},
    ]
    wavering_history = [
        {"theta_before": [0.0, 0.0], "theta_after": [0.3, 0.15]},
        {"theta_before": [0.3, 0.15], "theta_after": [-0.05, 0.12]},
        {"theta_before": [-0.05, 0.12], "theta_after": [0.1, -0.08]},
        {"theta_before": [0.1, -0.08], "theta_after": [-0.02, 0.2]},
    ]

    stable = analyzer.evaluate(history=stable_history, path=stable_path, dimensions=["extraversion", "agreeableness"], stop_threshold=0.7)
    wavering = analyzer.evaluate(history=wavering_history, path=wavering_path, dimensions=["extraversion", "agreeableness"], stop_threshold=0.7)

    assert stable["stability_score"] > wavering["stability_score"]


def test_stability_analyzer_penalizes_neutral_heavy_responses() -> None:
    analyzer = StabilityAnalyzer()
    neutral = analyzer.evaluate(
        history=[
            {"theta_before": [0.0], "theta_after": [0.05]},
            {"theta_before": [0.05], "theta_after": [0.08]},
            {"theta_before": [0.08], "theta_after": [0.1]},
        ],
        path=[
            {"dimension": "intellect", "response": 3, "keyed_response": 3},
            {"dimension": "intellect", "response": 3, "keyed_response": 3},
            {"dimension": "intellect", "response": 4, "keyed_response": 4},
        ],
        dimensions=["intellect"],
        stop_threshold=0.7,
    )
    decisive = analyzer.evaluate(
        history=[
            {"theta_before": [0.0], "theta_after": [0.05]},
            {"theta_before": [0.05], "theta_after": [0.08]},
            {"theta_before": [0.08], "theta_after": [0.1]},
        ],
        path=[
            {"dimension": "intellect", "response": 5, "keyed_response": 5},
            {"dimension": "intellect", "response": 4, "keyed_response": 4},
            {"dimension": "intellect", "response": 5, "keyed_response": 5},
        ],
        dimensions=["intellect"],
        stop_threshold=0.7,
    )
    assert decisive["stability_score"] > neutral["stability_score"]


def test_stability_analyzer_does_not_mark_all_neutral_sequence_as_stable() -> None:
    analyzer = StabilityAnalyzer()
    all_neutral = analyzer.evaluate(
        history=[
            {"theta_before": [0.0], "theta_after": [0.0]},
            {"theta_before": [0.0], "theta_after": [0.0]},
            {"theta_before": [0.0], "theta_after": [0.0]},
            {"theta_before": [0.0], "theta_after": [0.0]},
        ],
        path=[
            {"dimension": "intellect", "response": 3, "keyed_response": 3},
            {"dimension": "intellect", "response": 3, "keyed_response": 3},
            {"dimension": "agreeableness", "response": 3, "keyed_response": 3},
            {"dimension": "agreeableness", "response": 3, "keyed_response": 3},
        ],
        dimensions=["intellect", "agreeableness"],
        stop_threshold=0.7,
    )

    assert all_neutral["stability_score"] <= 0.45
    assert all_neutral["stability_ready"] is False
    assert all_neutral["stability_stage"] != "stable"


def test_stability_analyzer_does_not_mark_all_extreme_sequence_as_stable() -> None:
    analyzer = StabilityAnalyzer()
    all_extreme = analyzer.evaluate(
        history=[
            {"theta_before": [0.0], "theta_after": [0.18]},
            {"theta_before": [0.18], "theta_after": [0.34]},
            {"theta_before": [0.34], "theta_after": [0.47]},
            {"theta_before": [0.47], "theta_after": [0.58]},
        ],
        path=[
            {"dimension": "intellect", "response": 5, "keyed_response": 5},
            {"dimension": "intellect", "response": 5, "keyed_response": 5},
            {"dimension": "agreeableness", "response": 5, "keyed_response": 5},
            {"dimension": "agreeableness", "response": 5, "keyed_response": 5},
        ],
        dimensions=["intellect", "agreeableness"],
        stop_threshold=0.7,
    )

    assert all_extreme["components"]["response_diversity"] == 0.0
    assert all_extreme["stability_score"] <= 0.55
    assert all_extreme["stability_ready"] is False
    assert all_extreme["stability_stage"] != "stable"


def test_stability_analyzer_penalizes_fourteen_early_fives() -> None:
    analyzer = StabilityAnalyzer()
    path = []
    history = []
    theta = 0.0
    for index in range(14):
        next_theta = theta + 0.08
        history.append({"theta_before": [theta], "theta_after": [next_theta]})
        path.append(
            {
                "dimension": "intellect" if index % 2 == 0 else "agreeableness",
                "response": 5,
                "keyed_response": 5,
            }
        )
        theta = next_theta

    payload = analyzer.evaluate(
        history=history,
        path=path,
        dimensions=["intellect", "agreeableness"],
        stop_threshold=0.7,
    )

    assert payload["components"]["early_extreme_repetition"] is True
    assert payload["stability_score"] <= 0.35
    assert payload["stability_ready"] is False


def test_lower_response_weight_produces_smaller_theta_shift() -> None:
    first_router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl")
    second_router = AdaptiveMMPIRouter(device="cpu", scoring_model="binary_2pl")

    first_item = first_router.select_next_item()
    second_item = second_router.select_next_item()
    assert first_item is not None and second_item is not None
    assert first_item["id"] == second_item["id"]

    first_router.update_theta(str(first_item["id"]), 1.0, source="llm", response_weight=1.0)
    second_router.update_theta(str(second_item["id"]), 1.0, source="llm", response_weight=0.25)

    assert torch.norm(first_router.theta) > torch.norm(second_router.theta)


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
        param_mode="legacy",
        param_path=None,
    )

    assert sessions["param_mode"] == "legacy"
    assert sessions["key_aligned"] is False
    assert len(sessions["sessions"]) == 2
    assert {session["scoring_model"] for session in sessions["sessions"]} == {"binary_2pl", "grm"}
    assert all(session["answered_count"] == 3 for session in sessions["sessions"])
    assert all(len(session["path"]) == 3 for session in sessions["sessions"])
    assert all("classical_big5" in session for session in sessions["sessions"])
    assert all("irt_t_scores" in session for session in sessions["sessions"])
    assert all("stability_score" in session for session in sessions["sessions"])
    assert all("tendency_t_scores" not in session for session in sessions["sessions"])
    assert sessions["seed"] == 20260413


def test_simulation_seed_is_deterministic() -> None:
    first = run_session(
        persona=PERSONAS[0],
        scoring_model="binary_2pl",
        max_items=8,
        device="cpu",
        param_mode="keyed",
        param_path=None,
        response_style="wavering",
        seed=20260413,
    )
    second = run_session(
        persona=PERSONAS[0],
        scoring_model="binary_2pl",
        max_items=8,
        device="cpu",
        param_mode="keyed",
        param_path=None,
        response_style="wavering",
        seed=20260413,
    )

    assert first["style_seed"] == second["style_seed"]
    assert first["path"] == second["path"]


def test_different_response_styles_change_response_patterns() -> None:
    stable = run_session(
        persona=PERSONAS[0],
        scoring_model="binary_2pl",
        max_items=10,
        device="cpu",
        param_mode="keyed",
        param_path=None,
        response_style="stable",
        seed=20260413,
    )
    wavering = run_session(
        persona=PERSONAS[0],
        scoring_model="binary_2pl",
        max_items=10,
        device="cpu",
        param_mode="keyed",
        param_path=None,
        response_style="wavering",
        seed=20260413,
    )

    stable_responses = [step["response"] for step in stable["path"]]
    wavering_responses = [step["response"] for step in wavering["path"]]
    assert stable_responses != wavering_responses


def test_mixed_decisive_is_less_neutral_than_neutral_heavy() -> None:
    mixed = run_session(
        persona=PERSONAS[0],
        scoring_model="binary_2pl",
        max_items=20,
        device="cpu",
        param_mode="keyed",
        param_path=None,
        response_style="mixed_decisive",
        seed=20260413,
    )
    neutral_heavy = run_session(
        persona=PERSONAS[0],
        scoring_model="binary_2pl",
        max_items=20,
        device="cpu",
        param_mode="keyed",
        param_path=None,
        response_style="neutral_heavy",
        seed=20260413,
    )

    mixed_neutral = sum(1 for step in mixed["path"] if step["response"] == 3)
    neutral_heavy_neutral = sum(1 for step in neutral_heavy["path"] if step["response"] == 3)
    assert mixed_neutral < neutral_heavy_neutral


def test_param_source_resolves_keyed_mode() -> None:
    mode, path = resolve_param_source(param_mode="keyed")
    assert mode == "keyed"
    assert path.name == "mock_params_keyed.pt"


def test_benchmark_output_includes_param_metadata() -> None:
    report = run_benchmark(
        configs=DEFAULT_CONFIGS[:1],
        max_items=8,
        min_items=1,
        scoring_model="binary_2pl",
        device="cpu",
        param_mode="keyed",
        param_path=None,
    )

    assert report["param_mode"] == "keyed"
    assert report["key_aligned"] is True
    assert report["response_styles"] == [style.name for style in RESPONSE_STYLES]
    assert report["configs"][0]["personas"][0]["persona"]
    assert report["configs"][0]["personas"][0]["response_style"]
    assert "style_seed" in report["configs"][0]["personas"][0]
    assert "style_profile" in report["configs"][0]["personas"][0]
    assert "average_mean_standard_error" in report["configs"][0]
    assert "average_dimension_coverage" in report["configs"][0]
    assert "stability_score" in report["configs"][0]["personas"][0]
    assert "answered_count" in report["configs"][0]["personas"][0]["classical_big5"]["extraversion"]


def test_stable_benchmark_style_finishes_before_wavering() -> None:
    report = run_benchmark(
        configs=DEFAULT_CONFIGS[2:3],
        max_items=30,
        min_items=5,
        scoring_model="binary_2pl",
        device="cpu",
        param_mode="keyed",
        param_path=None,
        stop_stability_score=0.7,
    )
    sessions = report["configs"][0]["personas"]
    stable_sessions = [session for session in sessions if session["response_style"] == "stable"]
    wavering_sessions = [session for session in sessions if session["response_style"] == "wavering"]
    stable_average = sum(session["answered_count"] for session in stable_sessions) / len(stable_sessions)
    wavering_average = sum(session["answered_count"] for session in wavering_sessions) / len(wavering_sessions)

    assert stable_average < wavering_average


def test_mixed_decisive_finishes_later_than_stable_without_neutral_gate() -> None:
    report = run_benchmark(
        configs=DEFAULT_CONFIGS[:1],
        max_items=30,
        min_items=5,
        scoring_model="binary_2pl",
        device="cpu",
        param_mode="keyed",
        param_path=None,
        stop_stability_score=0.7,
    )
    sessions = report["configs"][0]["personas"]
    stable = next(session for session in sessions if session["persona"] == "balanced" and session["response_style"] == "stable")
    mixed = next(session for session in sessions if session["persona"] == "balanced" and session["response_style"] == "mixed_decisive")

    assert mixed["answered_count"] >= stable["answered_count"]
    assert mixed["stopped_by"] != "max_items_cap"


def test_param_mode_comparison_runs_both_tracks() -> None:
    comparison = build_comparison(max_items=8, min_items=1, scoring_model="binary_2pl", device="cpu")
    assert comparison["modes"] == ["legacy", "keyed"]
    assert set(comparison["benchmark"].keys()) == {"legacy", "keyed"}
    assert comparison["benchmark"]["legacy"]["key_aligned"] is False
    assert comparison["benchmark"]["keyed"]["key_aligned"] is True


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
