from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import AdaptiveMMPIRouter, ClassicalBigFiveScorer, resolve_param_source


SCRIPT_VERSION = "2026-04-12-param-mode-v1"


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


PERSONAS = [
    Persona("balanced", torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])),
    Persona("social_builder", torch.tensor([1.4, 0.9, 0.4, 0.3, 0.2])),
    Persona("focused_introvert", torch.tensor([-1.2, 0.2, 1.3, 0.4, 0.8])),
    Persona("stressed_explorer", torch.tensor([0.4, -0.2, -0.5, -1.4, 1.2])),
]


def clamp_likert(value: float) -> int:
    return max(1, min(5, int(round(value))))


def simulated_response(router: AdaptiveMMPIRouter, item_index: int, persona_theta: torch.Tensor) -> int:
    item = router.items[item_index]
    item_a = router.a[item_index].detach().cpu()
    item_b = router.b[item_index].detach().cpu()
    probability = torch.sigmoid(item_a @ persona_theta - item_b).item()
    if item.key < 0 and not router.key_aligned:
        probability = 1.0 - probability
    return clamp_likert(1.0 + 4.0 * probability)


def run_session(
    *,
    persona: Persona,
    scoring_model: str,
    max_items: int,
    device: str | None,
    param_mode: str | None,
    param_path: str | None,
) -> dict[str, object]:
    resolved_mode, resolved_path = resolve_param_source(param_mode=param_mode, param_path=param_path)
    router = AdaptiveMMPIRouter(scoring_model=scoring_model, device=device, param_path=resolved_path)
    classical = ClassicalBigFiveScorer(item_path=router.item_path)
    path: list[dict[str, object]] = []
    responses: dict[str, int] = {}

    for step in range(max_items):
        item = router.select_next_item()
        if item is None:
            break
        item_index = int(item["index"])
        response = simulated_response(router, item_index, persona.theta)
        record = router.answer_item(str(item["id"]), response)
        responses[str(item["id"])] = response
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
    return {
        "persona": persona.name,
        "scoring_model": scoring_model,
        "max_items": max_items,
        "param_mode": resolved_mode,
        "param_path": str(resolved_path),
        "key_aligned": router.key_aligned,
        "param_metadata": dict(router.param_metadata),
        "answered_count": router.answered_count,
        "mean_standard_error": uncertainty["mean_standard_error"],
        "confidence_ready": uncertainty["confidence_ready"],
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
        f"{session['persona']} / {session['scoring_model']} / "
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
) -> dict[str, object]:
    sessions: list[dict[str, object]] = []
    for persona in personas:
        for scoring_model in scoring_models:
            sessions.append(
                run_session(
                    persona=persona,
                    scoring_model=scoring_model,
                    max_items=max_items,
                    device=device,
                    param_mode=param_mode,
                    param_path=param_path,
                )
            )
    first = sessions[0]
    return {
        "experiment": "simulation_matrix",
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "param_mode": first["param_mode"],
        "param_path": first["param_path"],
        "key_aligned": first["key_aligned"],
        "param_metadata": dict(first["param_metadata"]),
        "max_items": max_items,
        "scoring_models": list(scoring_models),
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
    sessions = run_matrix(
        personas=PERSONAS,
        scoring_models=models,
        max_items=args.max_items,
        device=args.device,
        param_mode=args.param_mode,
        param_path=args.param_path,
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
