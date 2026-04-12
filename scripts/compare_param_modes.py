from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_stopping_rules import DEFAULT_CONFIGS, run_benchmark
from scripts.simulate_adaptive_sessions import PERSONAS, run_matrix


SCRIPT_VERSION = "2026-04-12-param-mode-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run legacy vs keyed parameter comparisons for CAT-Psych.")
    parser.add_argument("--max-items", type=int, default=50, help="Hard item cap for each comparison run.")
    parser.add_argument("--min-items", type=int, default=8, help="Minimum items before early stopping is allowed.")
    parser.add_argument(
        "--model",
        choices=["binary_2pl", "grm"],
        default="binary_2pl",
        help="Scoring model to compare.",
    )
    parser.add_argument("--device", default=None, help="Torch device override, such as cpu or cuda.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def build_comparison(*, max_items: int, min_items: int, scoring_model: str, device: str | None) -> dict[str, object]:
    benchmark = {
        mode: run_benchmark(
            configs=DEFAULT_CONFIGS,
            max_items=max_items,
            min_items=min_items,
            scoring_model=scoring_model,
            device=device,
            param_mode=mode,
            param_path=None,
        )
        for mode in ("legacy", "keyed")
    }
    simulation = {
        mode: run_matrix(
            personas=PERSONAS,
            scoring_models=[scoring_model],
            max_items=min(max_items, 12),
            device=device,
            param_mode=mode,
            param_path=None,
        )
        for mode in ("legacy", "keyed")
    }
    return {
        "experiment": "param_mode_comparison",
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "scoring_model": scoring_model,
        "modes": ["legacy", "keyed"],
        "benchmark": benchmark,
        "simulation": simulation,
    }


def main() -> None:
    args = parse_args()
    comparison = build_comparison(
        max_items=args.max_items,
        min_items=args.min_items,
        scoring_model=args.model,
        device=args.device,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved comparison JSON to {args.output}")
        return
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
