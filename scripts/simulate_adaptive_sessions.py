from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import AdaptiveMMPIRouter, ClassicalBigFiveScorer


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
    if item.key < 0:
        probability = 1.0 - probability
    return clamp_likert(1.0 + 4.0 * probability)


def run_session(
    *,
    persona: Persona,
    scoring_model: str,
    max_items: int,
    device: str | None,
) -> dict[str, object]:
    router = AdaptiveMMPIRouter(scoring_model=scoring_model, device=device)
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

    return {
        "persona": persona.name,
        "scoring_model": scoring_model,
        "max_items": max_items,
        "answered_count": router.answered_count,
        "trait_estimates": router.trait_estimates(),
        "tendency_t_scores": router.tendency_t_scores(),
        "classical_big5": classical.score(responses),
        "path": path,
    }


def summarize_session(session: dict[str, object]) -> str:
    traits = session["trait_estimates"]
    t_scores = session["tendency_t_scores"]
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
) -> list[dict[str, object]]:
    sessions: list[dict[str, object]] = []
    for persona in personas:
        for scoring_model in scoring_models:
            sessions.append(
                run_session(
                    persona=persona,
                    scoring_model=scoring_model,
                    max_items=max_items,
                    device=device,
                )
            )
    return sessions


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
    )

    for session in sessions:
        print(summarize_session(session))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(sessions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved simulation JSON to {args.output}")


if __name__ == "__main__":
    main()
