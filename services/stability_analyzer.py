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

        config = self.config
        score = (
            config.volatility_weight * volatility
            + config.consistency_weight * consistency
            + config.decisiveness_weight * decisiveness
        )
        if decisiveness < 0.4:
            score = min(score, 0.45)
        score = max(0.0, min(1.0, score))
        return {
            "stability_score": score,
            "stability_ready": bool(score >= stop_threshold),
            "stability_stage": self._stage_for(score, stop_threshold),
            "components": {
                "recent_volatility": volatility,
                "dimension_consistency": consistency,
                "response_decisiveness": decisiveness,
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
    def _stage_for(score: float, threshold: float) -> str:
        if score >= threshold:
            return "stable"
        if score >= max(0.45, threshold - 0.15):
            return "mixed"
        return "unstable"
