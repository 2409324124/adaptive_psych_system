from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import AdaptiveMMPIRouter, ClassicalBigFiveScorer, resolve_param_source


SCRIPT_VERSION = "2026-04-13-response-style-v2"
DEFAULT_SEED = 20260413


TRAIT_ORDER = [
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "emotional_stability",
    "intellect",
]


@dataclass(frozen=True)
class Persona:
    name: str
    theta: torch.Tensor


@dataclass(frozen=True)
class ResponseStyle:
    name: str
    neutral_bias: float
    flip_rate: float
    noise_scale: float
    consistency_strength: float
    seed_offset: int
    decisive_flip_strength: float = 0.0


PERSONAS = [
    Persona("balanced", torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])),
    Persona("social_builder", torch.tensor([1.4, 0.9, 0.4, 0.3, 0.2])),
    Persona("focused_introvert", torch.tensor([-1.2, 0.2, 1.3, 0.4, 0.8])),
    Persona("stressed_explorer", torch.tensor([0.4, -0.2, -0.5, -1.4, 1.2])),
]

RESPONSE_STYLES = [
    ResponseStyle(
        "stable",
        neutral_bias=0.03,
        flip_rate=0.04,
        noise_scale=0.05,
        consistency_strength=0.82,
        seed_offset=11,
    ),
    ResponseStyle(
        "wavering",
        neutral_bias=0.08,
        flip_rate=0.30,
        noise_scale=0.16,
        consistency_strength=0.22,
        seed_offset=29,
    ),
    ResponseStyle(
        "neutral_heavy",
        neutral_bias=0.48,
        flip_rate=0.10,
        noise_scale=0.08,
        consistency_strength=0.72,
        seed_offset=41,
    ),
    ResponseStyle(
        "mixed_decisive",
        neutral_bias=0.05,
        flip_rate=0.38,
        noise_scale=0.14,
        consistency_strength=0.18,
        seed_offset=53,
        decisive_flip_strength=0.22,
    ),
]


def clamp_likert(value: float) -> int:
    return max(1, min(5, int(round(value))))


def clamp_probability(value: float) -> float:
    return max(0.01, min(0.99, value))


def derive_style_seed(*, seed: int, persona_name: str, response_style: ResponseStyle) -> int:
    payload = f"{seed}:{persona_name}:{response_style.name}:{response_style.seed_offset}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


def style_by_name(name: str) -> ResponseStyle:
    for style in RESPONSE_STYLES:
        if style.name == name:
            return style
    raise KeyError(f"Unknown response style: {name}")


def simulated_response(
    router: AdaptiveMMPIRouter,
    item_index: int,
    persona_theta: torch.Tensor,
    *,
    response_style: ResponseStyle,
    step: int = 0,
    seed: int,
    persona_name: str,
    previous_dimension_response: int | None = None,
) -> int:
    item = router.items[item_index]
    item_a = router.a[item_index].detach().cpu()
    item_b = router.b[item_index].detach().cpu()
    probability = torch.sigmoid(item_a @ persona_theta - item_b).item()
    if item.key < 0 and not router.key_aligned:
        probability = 1.0 - probability

    style_seed = derive_style_seed(seed=seed, persona_name=persona_name, response_style=response_style)
    rng = random.Random(style_seed + step * 7919 + item_index * 104729)

    probability = clamp_probability(probability + rng.gauss(0.0, response_style.noise_scale))

    if previous_dimension_response is not None:
        previous_target = (previous_dimension_response - 1.0) / 4.0
        probability = (
            response_style.consistency_strength * previous_target
            + (1.0 - response_style.consistency_strength) * probability
        )

    if rng.random() < response_style.flip_rate:
        if response_style.decisive_flip_strength > 0.0:
            push = response_style.decisive_flip_strength if probability >= 0.5 else -response_style.decisive_flip_strength
            probability = clamp_probability((1.0 - probability) + push)
        else:
            probability = 1.0 - probability

    if response_style.neutral_bias > 0.0:
        probability = (1.0 - response_style.neutral_bias) * probability + response_style.neutral_bias * 0.5

    if response_style.decisive_flip_strength > 0.0:
        distance = probability - 0.5
        probability = clamp_probability(0.5 + distance * (1.0 + response_style.decisive_flip_strength))

    return clamp_likert(1.0 + 4.0 * probability)


def run_session(
    *,
    persona: Persona,
    scoring_model: str,
    max_items: int,
    device: str | None,
    param_mode: str | None,
    param_path: str | None,
    response_style: str = "stable",
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    resolved_mode, resolved_path = resolve_param_source(param_mode=param_mode, param_path=param_path)
    router = AdaptiveMMPIRouter(scoring_model=scoring_model, device=device, param_path=resolved_path)
    classical = ClassicalBigFiveScorer(item_path=router.item_path)
    path: list[dict[str, object]] = []
    responses: dict[str, int] = {}
    style = style_by_name(response_style)
    style_seed = derive_style_seed(seed=seed, persona_name=persona.name, response_style=style)
    previous_responses_by_dimension: dict[str, int] = {}

    for step in range(max_items):
        item = router.select_next_item()
        if item is None:
            break
        item_index = int(item["index"])
        response = simulated_response(
            router,
            item_index,
            persona.theta,
            response_style=style,
            step=step,
            seed=seed,
            persona_name=persona.name,
            previous_dimension_response=previous_responses_by_dimension.get(str(item["dimension"])),
        )
        record = router.answer_item(str(item["id"]), response)
        responses[str(item["id"])] = response
        previous_responses_by_dimension[str(item["dimension"])] = response
        path.append(
            {
                "step": step + 1,
                "item_id": item["id"],
                "dimension": item["dimension"],
                "key": item["key"],
                "response": response,
                "theta_after": record["theta_after"],
            }
        )

    classical_scores = classical.score(responses)
    dimension_counts = router.dimension_answer_counts()
    uncertainty = router.uncertainty_summary()
    stability_score = 0.0
    stability_stage = "unstable"
    if router.history:
        from services.stability_analyzer import StabilityAnalyzer

        path_records = [
            {
                "step": record["step"],
                "item_id": record["item_id"],
                "dimension": record["dimension"],
                "key": record["key"],
                "response": record["response"],
                "keyed_response": router.history[index]["keyed_response"],
                "theta_after": record["theta_after"],
            }
            for index, record in enumerate(path)
        ]
        stability = StabilityAnalyzer().evaluate(
            history=router.history,
            path=path_records,
            dimensions=router.dimensions,
            stop_threshold=0.7,
        )
        stability_score = float(stability["stability_score"])
        stability_stage = str(stability["stability_stage"])
    return {
        "persona": persona.name,
        "response_style": response_style,
        "style_seed": style_seed,
        "style_profile": asdict(style),
        "scoring_model": scoring_model,
        "max_items": max_items,
        "seed": seed,
        "param_mode": resolved_mode,
        "param_path": str(resolved_path),
        "key_aligned": router.key_aligned,
        "param_metadata": dict(router.param_metadata),
        "answered_count": router.answered_count,
        "mean_standard_error": uncertainty["mean_standard_error"],
        "confidence_ready": uncertainty["confidence_ready"],
        "stability_score": stability_score,
        "stability_stage": stability_stage,
        "dimension_answer_counts": dimension_counts,
        "trait_estimates": router.trait_estimates(),
        "irt_t_scores": router.tendency_t_scores(),
        "classical_big5": classical_scores,
        "path": path,
    }


def summarize_session(session: dict[str, object]) -> str:
    traits = session["trait_estimates"]
    t_scores = session["irt_t_scores"]
    classical = session["classical_big5"]
    assert isinstance(traits, dict)
    assert isinstance(t_scores, dict)
    assert isinstance(classical, dict)
    trait_bits = ", ".join(f"{trait}={traits[trait]:+.2f}" for trait in TRAIT_ORDER)
    t_bits = ", ".join(f"{trait}={t_scores[trait]:.1f}" for trait in TRAIT_ORDER)
    classical_bits = ", ".join(
        f"{trait}={classical[trait]['tendency_t_score']:.1f}"
        if classical[trait]["tendency_t_score"] is not None
        else f"{trait}=NA"
        for trait in TRAIT_ORDER
    )
    return (
        f"{session['persona']} / {session['response_style']} / {session['scoring_model']} / "
        f"n={session['answered_count']}\n"
        f"  theta: {trait_bits}\n"
        f"  IRT T: {t_bits}\n"
        f"  classical T on routed items: {classical_bits}"
    )


def run_matrix(
    *,
    personas: Iterable[Persona],
    scoring_models: Iterable[str],
    max_items: int,
    device: str | None,
    param_mode: str | None,
    param_path: str | None,
    response_styles: Iterable[ResponseStyle] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    sessions: list[dict[str, object]] = []
    active_styles = list(response_styles or [style_by_name("stable")])
    for persona in personas:
        for style in active_styles:
            for scoring_model in scoring_models:
                sessions.append(
                    run_session(
                        persona=persona,
                        scoring_model=scoring_model,
                        max_items=max_items,
                        device=device,
                        param_mode=param_mode,
                        param_path=param_path,
                        response_style=style.name,
                        seed=seed,
                    )
                )
    first = sessions[0]
    return {
        "experiment": "simulation_matrix",
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "param_mode": first["param_mode"],
        "param_path": first["param_path"],
        "key_aligned": first["key_aligned"],
        "param_metadata": dict(first["param_metadata"]),
        "max_items": max_items,
        "scoring_models": list(scoring_models),
        "response_styles": [style.name for style in active_styles],
        "sessions": sessions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small CAT-Psych adaptive-routing simulations.")
    parser.add_argument("--max-items", type=int, default=12, help="Maximum items per simulated session.")
    parser.add_argument(
        "--model",
        choices=["binary_2pl", "grm", "both"],
        default="both",
        help="Scoring model to simulate.",
    )
    parser.add_argument(
        "--response-style",
        choices=["stable", "wavering", "neutral_heavy", "mixed_decisive", "all"],
        default="stable",
        help="Response style to simulate.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic seed for response-style simulation.")
    parser.add_argument("--device", default=None, help="Torch device override, such as cpu or cuda.")
    parser.add_argument(
        "--param-mode",
        choices=["legacy", "keyed"],
        default=None,
        help="Named parameter mode. Keeps old defaults unless explicitly set.",
    )
    parser.add_argument("--param-path", default=None, help="Optional torch parameter file path override.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = ["binary_2pl", "grm"] if args.model == "both" else [args.model]
    active_styles = RESPONSE_STYLES if args.response_style == "all" else [style_by_name(args.response_style)]
    sessions = run_matrix(
        personas=PERSONAS,
        scoring_models=models,
        max_items=args.max_items,
        device=args.device,
        param_mode=args.param_mode,
        param_path=args.param_path,
        response_styles=active_styles,
        seed=args.seed,
    )

    print(
        f"PARAMS mode={sessions['param_mode']} key_aligned={sessions['key_aligned']} "
        f"path={sessions['param_path']}"
    )
    for session in sessions["sessions"]:
        print(summarize_session(session))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(sessions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved simulation JSON to {args.output}")


if __name__ == "__main__":
    main()
