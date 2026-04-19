from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import DISCLAIMER_ASCII, TRAIT_ORDER
from services import AssessmentSession



def parse_demo_responses(value: str | None) -> list[int] | None:
    if value is None:
        return None
    responses = [int(part.strip()) for part in value.split(",") if part.strip()]
    for response in responses:
        if response not in {1, 2, 3, 4, 5}:
            raise ValueError("--demo-responses values must be integers from 1 to 5.")
    return responses


def ask_likert(prompt: str) -> int:
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError as exc:
            raise KeyboardInterrupt from exc
        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        try:
            response = int(raw)
        except ValueError:
            print("Please enter 1, 2, 3, 4, or 5. Enter q to quit.")
            continue
        if response in {1, 2, 3, 4, 5}:
            return response
        print("Please enter 1, 2, 3, 4, or 5. Enter q to quit.")


def safe_print(text: object = "") -> None:
    value = str(text)
    print(value.encode("ascii", errors="replace").decode("ascii"))


def printable_disclaimer() -> str:
    return DISCLAIMER_ASCII


def format_t_scores(scores: dict[str, float]) -> str:
    return "\n".join(f"  {trait}: {scores[trait]:.1f}" for trait in TRAIT_ORDER)


def format_classical_scores(scores: dict[str, dict[str, float | int | None]]) -> str:
    lines = []
    for trait in TRAIT_ORDER:
        score = scores[trait]
        t_score = score["tendency_t_score"]
        answered_count = score["answered_count"]
        if t_score is None:
            lines.append(f"  {trait}: NA (answered={answered_count})")
        else:
            lines.append(f"  {trait}: {t_score:.1f} (answered={answered_count})")
    return "\n".join(lines)


def run_assessment(
    *,
    scoring_model: str,
    max_items: int,
    device: str | None,
    demo_responses: list[int] | None = None,
) -> dict[str, object]:
    session = AssessmentSession(
        scoring_model=scoring_model,
        max_items=max_items,
        device=device,
        min_items=1,
        coverage_min_per_dimension=0,
    )
    disclaimer = printable_disclaimer()

    safe_print("\nCAT-Psych Engine CLI")
    safe_print(disclaimer)
    safe_print("\nScale: 1=Very inaccurate, 2=Moderately inaccurate, 3=Neutral, 4=Moderately accurate, 5=Very accurate")
    safe_print("Enter q to quit.\n")

    step = 0
    while not session.is_complete:
        question = session.next_question()
        if question is None:
            break

        step += 1
        item_id = str(question["item_id"])

        safe_print(f"[{step}/{max_items}] {question['text']}")
        safe_print(f"Dimension hint: {question['dimension']} | key={question['key']} | model={scoring_model}")

        if demo_responses is None:
            response = ask_likert("Your response (1-5): ")
        else:
            response = demo_responses[(step - 1) % len(demo_responses)]
            safe_print(f"Demo response: {response}")

        session.submit_response(item_id, response)
        safe_print()

    result = session.result()
    result["answered_count"] = session.answered_count

    safe_print("Results")
    safe_print("IRT tendency T scores:")
    safe_print(format_t_scores(result["irt_t_scores"]))
    safe_print("\nClassical Big Five T scores on answered items:")
    safe_print(format_classical_scores(result["classical_big5"]))
    safe_print("\nReminder:")
    safe_print(disclaimer)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a manual CAT-Psych engine assessment in the terminal.")
    parser.add_argument("--model", choices=["binary_2pl", "grm"], default="binary_2pl")
    parser.add_argument("--max-items", type=int, default=12)
    parser.add_argument("--device", default=None, help="Torch device override, such as cpu or cuda.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--demo-responses",
        default=None,
        help="Comma-separated 1-5 responses for non-interactive smoke tests, e.g. 5,4,3,2,1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo_responses = parse_demo_responses(args.demo_responses)
    try:
        result = run_assessment(
            scoring_model=args.model,
            max_items=args.max_items,
            device=args.device,
            demo_responses=demo_responses,
        )
    except KeyboardInterrupt:
        safe_print("\nAssessment stopped.")
        return

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        safe_print(f"\nSaved assessment JSON to {args.output}")


if __name__ == "__main__":
    main()
