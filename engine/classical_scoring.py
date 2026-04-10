from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassicalItem:
    item_id: str
    dimension: str
    key: int


class ClassicalBigFiveScorer:
    def __init__(self, *, item_path: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        self.item_path = Path(item_path) if item_path is not None else root / "data" / "ipip_items.json"
        self.items, self.dimensions = self._load_items()
        self.item_by_id = {item.item_id: item for item in self.items}

    def _load_items(self) -> tuple[list[ClassicalItem], list[str]]:
        with self.item_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        items = [
            ClassicalItem(
                item_id=item["id"],
                dimension=item["dimension"],
                key=int(item.get("key", 1)),
            )
            for item in payload["items"]
        ]
        return items, list(payload["dimensions"])

    @staticmethod
    def keyed_score(response: int, key: int) -> int:
        if response not in {1, 2, 3, 4, 5}:
            raise ValueError("Likert response must be an integer from 1 to 5.")
        if key not in {-1, 1}:
            raise ValueError("Classical item key must be either 1 or -1.")
        return response if key == 1 else 6 - response

    @staticmethod
    def centered_score(raw_mean: float) -> float:
        return (raw_mean - 3.0) / 2.0

    @staticmethod
    def tendency_t_score(centered_score: float) -> float:
        return 50.0 + 10.0 * centered_score

    def score(self, responses: dict[str, int]) -> dict[str, dict[str, float | int | None]]:
        buckets: dict[str, list[int]] = {dimension: [] for dimension in self.dimensions}
        for item_id, response in responses.items():
            item = self.item_by_id[item_id]
            buckets[item.dimension].append(self.keyed_score(response, item.key))

        results: dict[str, dict[str, float | int | None]] = {}
        for dimension, scores in buckets.items():
            if not scores:
                results[dimension] = {
                    "answered_count": 0,
                    "raw_mean": None,
                    "centered_score": None,
                    "tendency_t_score": None,
                }
                continue

            raw_mean = sum(scores) / len(scores)
            centered = self.centered_score(raw_mean)
            results[dimension] = {
                "answered_count": len(scores),
                "raw_mean": raw_mean,
                "centered_score": centered,
                "tendency_t_score": self.tendency_t_score(centered),
            }
        return results

    def score_complete(self, responses: dict[str, int]) -> dict[str, dict[str, float | int | None]]:
        missing = {item.item_id for item in self.items}.difference(responses)
        if missing:
            raise ValueError(f"Missing responses for {len(missing)} items.")
        return self.score(responses)
