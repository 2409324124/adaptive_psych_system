from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class StabilityConfig:
    recent_window: int = 5
    volatility_scale: float = 0.6
    consistency_weight: float = 0.45
    volatility_weight: float = 0.3
    decisiveness_weight: float = 0.25
    early_extreme_window: int = 14
    early_extreme_cap: float = 0.35
    low_diversity_cap: float = 0.55
    medium_diversity_cap: float = 0.65


class StabilityAnalyzer:
    def __init__(self, config: StabilityConfig | None = None) -> None:
        self.config = config or StabilityConfig()

    def evaluate(
        self,
        *,
        history: list[dict[str, object]],
        path: list[dict[str, object]],
        dimensions: list[str],
        stop_threshold: float,
    ) -> dict[str, object]:
        if not history or not path:
            return {
                "stability_score": 0.0,
                "stability_ready": False,
                "stability_stage": "unstable",
                "components": {
                    "recent_volatility": 0.0,
                    "dimension_consistency": 0.0,
                    "response_decisiveness": 0.0,
                },
            }

        volatility = self._recent_volatility(history)
        consistency = self._dimension_consistency(path, dimensions)
        decisiveness = self._response_decisiveness(path)
        diversity = self._response_diversity(path)
        early_extreme_repetition = self._has_early_extreme_repetition(path)

        config = self.config
        score = (
            config.volatility_weight * volatility
            + config.consistency_weight * consistency
            + config.decisiveness_weight * decisiveness
        )
        if decisiveness < 0.4:
            score = min(score, 0.45)
        if early_extreme_repetition:
            score = min(score, config.early_extreme_cap)
        if diversity < 0.35:
            score = min(score, config.low_diversity_cap)
        elif diversity < 0.5:
            score = min(score, config.medium_diversity_cap)
        score = max(0.0, min(1.0, score))
        return {
            "stability_score": score,
            "stability_ready": bool(score >= stop_threshold),
            "stability_stage": self._stage_for(score, stop_threshold),
            "components": {
                "recent_volatility": volatility,
                "dimension_consistency": consistency,
                "response_decisiveness": decisiveness,
                "response_diversity": diversity,
                "early_extreme_repetition": early_extreme_repetition,
            },
        }

    def _recent_volatility(self, history: list[dict[str, object]]) -> float:
        deltas: list[float] = []
        for record in history[-self.config.recent_window :]:
            theta_before = [float(value) for value in record["theta_before"]]
            theta_after = [float(value) for value in record["theta_after"]]
            squared = [(after - before) ** 2 for before, after in zip(theta_before, theta_after, strict=True)]
            deltas.append(sqrt(sum(squared)))

        if not deltas:
            return 0.0
        mean_delta = sum(deltas) / len(deltas)
        return max(0.0, min(1.0, 1.0 - (mean_delta / self.config.volatility_scale)))

    def _dimension_consistency(self, path: list[dict[str, object]], dimensions: list[str]) -> float:
        by_dimension: dict[str, list[float]] = {dimension: [] for dimension in dimensions}
        for step in path:
            by_dimension[str(step["dimension"])].append((float(step["keyed_response"]) - 1.0) / 4.0)

        scores: list[float] = []
        for responses in by_dimension.values():
            if len(responses) < 2:
                continue
            mean_value = sum(responses) / len(responses)
            variance = sum((value - mean_value) ** 2 for value in responses) / len(responses)
            scores.append(max(0.0, min(1.0, 1.0 - (variance / 0.18))))

        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @staticmethod
    def _response_decisiveness(path: list[dict[str, object]]) -> float:
        responses = [int(step["response"]) for step in path]
        if not responses:
            return 0.0
        neutral_count = sum(1 for response in responses if response == 3)
        return max(0.0, min(1.0, 1.0 - (neutral_count / len(responses))))

    @staticmethod
    def _response_diversity(path: list[dict[str, object]]) -> float:
        responses = [int(step["response"]) for step in path]
        if not responses:
            return 0.0

        counts: dict[int, int] = {}
        for response in responses:
            counts[response] = counts.get(response, 0) + 1

        unique_count = len(counts)
        dominant_ratio = max(counts.values()) / len(responses)
        all_extreme = unique_count == 1 and next(iter(counts)) in {1, 5}
        polarized_dual = set(counts).issubset({1, 5}) and unique_count <= 2 and dominant_ratio >= 0.75

        if all_extreme:
            return 0.0
        if polarized_dual:
            return 0.3

        diversity_score = min(1.0, max(0.0, (unique_count - 1) / 3))
        balance_score = max(0.0, min(1.0, 1.0 - dominant_ratio))
        return max(0.0, min(1.0, 0.65 * diversity_score + 0.35 * balance_score))

    def _has_early_extreme_repetition(self, path: list[dict[str, object]]) -> bool:
        if len(path) < self.config.early_extreme_window:
            return False
        first_window = [int(step["response"]) for step in path[: self.config.early_extreme_window]]
        return all(response == 5 for response in first_window) or all(response == 1 for response in first_window)

    @staticmethod
    def _stage_for(score: float, threshold: float) -> str:
        if score >= threshold:
            return "stable"
        if score >= max(0.45, threshold - 0.15):
            return "mixed"
        return "unstable"
