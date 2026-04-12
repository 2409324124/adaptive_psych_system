from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import resolve_param_source
from services import AssessmentSession
from scripts.simulate_adaptive_sessions import PERSONAS, RESPONSE_STYLES, derive_style_seed, simulated_response


SCRIPT_VERSION = "2026-04-13-response-style-v2"
DEFAULT_SEED = 20260413


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    coverage_min_per_dimension: int
    stop_mean_standard_error: float


DEFAULT_CONFIGS = [
    BenchmarkConfig("coverage=2,se=0.85", coverage_min_per_dimension=2, stop_mean_standard_error=0.85),
    BenchmarkConfig("coverage=2,se=0.65", coverage_min_per_dimension=2, stop_mean_standard_error=0.65),
    BenchmarkConfig("coverage=3,se=0.65", coverage_min_per_dimension=3, stop_mean_standard_error=0.65),
]


def run_benchmark(
    *,
    configs: list[BenchmarkConfig],
    max_items: int,
    min_items: int,
    scoring_model: str,
    device: str | None,
    param_mode: str | None,
    param_path: str | None,
    stop_stability_score: float = 0.7,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    resolved_mode, resolved_path = resolve_param_source(param_mode=param_mode, param_path=param_path, root=ROOT)
    results: list[dict[str, object]] = []
    key_aligned: bool | None = None
    param_metadata: dict[str, object] = {}
    for config in configs:
        sessions: list[dict[str, object]] = []
        for persona in PERSONAS:
            for response_style in RESPONSE_STYLES:
                session = AssessmentSession(
                    scoring_model=scoring_model,
                    max_items=max_items,
                    min_items=min_items,
                    coverage_min_per_dimension=config.coverage_min_per_dimension,
                    stop_mean_standard_error=config.stop_mean_standard_error,
                    stop_stability_score=stop_stability_score,
                    device=device,
                    param_mode=resolved_mode,
                    param_path=str(resolved_path),
                )
                key_aligned = session.router.key_aligned
                param_metadata = dict(session.router.param_metadata)
                previous_responses_by_dimension: dict[str, int] = {}
                style_seed = derive_style_seed(seed=seed, persona_name=persona.name, response_style=response_style)

                while not session.is_complete:
                    item = session.next_question()
                    if item is None:
                        break
                    item_index = session.router._index_for_item_id(str(item["item_id"]))
                    response = simulated_response(
                        session.router,
                        item_index,
                        persona.theta,
                        response_style=response_style,
                        step=session.answered_count,
                        seed=seed,
                        persona_name=persona.name,
                        previous_dimension_response=previous_responses_by_dimension.get(str(item["dimension"])),
                    )
                    session.submit_response(str(item["item_id"]), response)
                    previous_responses_by_dimension[str(item["dimension"])] = response

                result = session.result()
                sessions.append(
                    {
                        "persona": persona.name,
                        "response_style": response_style.name,
                        "style_seed": style_seed,
                        "style_profile": {
                            "neutral_bias": response_style.neutral_bias,
                            "flip_rate": response_style.flip_rate,
                            "noise_scale": response_style.noise_scale,
                            "consistency_strength": response_style.consistency_strength,
                            "seed_offset": response_style.seed_offset,
                            "decisive_flip_strength": response_style.decisive_flip_strength,
                        },
                        "answered_count": result["progress"]["answered"],
                        "mean_standard_error": result["uncertainty"]["mean_standard_error"],
                        "dimension_answer_counts": result["dimension_answer_counts"],
                        "confidence_ready": result["uncertainty"]["confidence_ready"],
                        "stability_score": result["stability"]["stability_score"],
                        "stability_stage": result["stability"]["stability_stage"],
                        "stopped_by": result["progress"]["stopped_by"],
                        "irt_t_scores": result["irt_t_scores"],
                        "classical_big5": result["classical_big5"],
                    }
                )

        average_answered = sum(session["answered_count"] for session in sessions) / len(sessions)
        average_mean_standard_error = sum(session["mean_standard_error"] for session in sessions) / len(sessions)
        dimensions = list(sessions[0]["dimension_answer_counts"].keys())
        average_dimension_coverage = {
            dimension: sum(session["dimension_answer_counts"][dimension] for session in sessions) / len(sessions)
            for dimension in dimensions
        }
        results.append(
            {
                "config": config.name,
                "stopping_rule": {
                    "coverage_min_per_dimension": config.coverage_min_per_dimension,
                    "stop_mean_standard_error": config.stop_mean_standard_error,
                    "stop_stability_score": stop_stability_score,
                    "min_items": min_items,
                    "max_items": max_items,
                },
                "average_answered_count": average_answered,
                "average_mean_standard_error": average_mean_standard_error,
                "average_dimension_coverage": average_dimension_coverage,
                "personas": sessions,
            }
        )
    return {
        "experiment": "benchmark_stopping_rules",
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "scoring_model": scoring_model,
        "param_mode": resolved_mode,
        "param_path": str(resolved_path),
        "key_aligned": bool(key_aligned),
        "param_metadata": param_metadata,
        "response_styles": [style.name for style in RESPONSE_STYLES],
        "configs": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CAT-Psych stopping rules across simulated personas.")
    parser.add_argument("--max-items", type=int, default=50, help="Hard item cap for each simulated session.")
    parser.add_argument("--min-items", type=int, default=8, help="Minimum items before early stopping is allowed.")
    parser.add_argument("--stop-stability-score", type=float, default=0.7, help="Stability score threshold required for early stopping.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic seed for response-style simulation.")
    parser.add_argument(
        "--model",
        choices=["binary_2pl", "grm"],
        default="binary_2pl",
        help="Scoring model to benchmark.",
    )
    parser.add_argument(
        "--param-mode",
        choices=["legacy", "keyed"],
        default=None,
        help="Named parameter mode. Keeps old defaults unless explicitly set.",
    )
    parser.add_argument("--param-path", default=None, help="Optional torch parameter file path override.")
    parser.add_argument("--device", default=None, help="Torch device override, such as cpu or cuda.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def print_report(results: dict[str, object]) -> None:
    print(
        f"PARAMS mode={results['param_mode']} key_aligned={results['key_aligned']} "
        f"path={results['param_path']}"
    )
    for block in results["configs"]:
        print(f"CONFIG {block['config']}")
        for session in block["personas"]:
            print(
                f"  {session['persona']} / {session['response_style']}: answered={session['answered_count']}, "
                f"mean_se={session['mean_standard_error']:.3f}, "
                f"stability={session['stability_score']:.3f} ({session['stability_stage']}), "
                f"stop={session['stopped_by']}, counts={session['dimension_answer_counts']}"
            )
        print(
            f"  average_answered={block['average_answered_count']:.2f}, "
            f"average_mean_se={block['average_mean_standard_error']:.3f}"
        )
        print()


def main() -> None:
    args = parse_args()
    results = run_benchmark(
        configs=DEFAULT_CONFIGS,
        max_items=args.max_items,
        min_items=args.min_items,
        scoring_model=args.model,
        device=args.device,
        param_mode=args.param_mode,
        param_path=args.param_path,
        stop_stability_score=args.stop_stability_score,
        seed=args.seed,
    )
    print_report(results)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved benchmark JSON to {args.output}")


if __name__ == "__main__":
    main()
