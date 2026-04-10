from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from engine import AdaptiveMMPIRouter, ClassicalBigFiveScorer


DISCLAIMER = "\u672c\u7cfb\u7edf\u4ec5\u4f5c\u4e3a\u5fc3\u7406\u7279\u8d28\u7b5b\u67e5\u4e0e\u8f85\u52a9\u53c2\u8003\u5de5\u5177\uff0c\u7edd\u5bf9\u4e0d\u53ef\u66ff\u4ee3\u4e13\u4e1a\u7cbe\u795e\u79d1\u4e34\u5e8a\u8bca\u65ad\u3002"


@dataclass
class AssessmentSession:
    scoring_model: str = "binary_2pl"
    max_items: int = 12
    device: str | None = None
    coverage_min_per_dimension: int = 2
    session_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        self.router = AdaptiveMMPIRouter(
            scoring_model=self.scoring_model,
            device=self.device,
            coverage_min_per_dimension=self.coverage_min_per_dimension,
        )
        self.classical = ClassicalBigFiveScorer(item_path=self.router.item_path)
        self.responses: dict[str, int] = {}
        self.path: list[dict[str, object]] = []

    @property
    def answered_count(self) -> int:
        return self.router.answered_count

    @property
    def is_complete(self) -> bool:
        return self.answered_count >= self.max_items or self.router.remaining_count <= 0

    def progress(self) -> dict[str, int | bool]:
        return {
            "answered": self.answered_count,
            "max_items": self.max_items,
            "remaining": max(0, self.max_items - self.answered_count),
            "complete": self.is_complete,
        }

    def next_question(self) -> dict[str, object] | None:
        if self.is_complete:
            return None
        item = self.router.select_next_item()
        if item is None:
            return None
        return {
            "session_id": self.session_id,
            "item_id": item["id"],
            "text": item["text"],
            "response_scale": item["response_scale"],
            "progress": self.progress(),
        }

    def submit_response(self, item_id: str, response: int) -> dict[str, object]:
        if self.is_complete:
            raise ValueError("This assessment session is already complete.")
        current = self.router.select_next_item()
        if current is None:
            raise ValueError("No active item is available.")
        if item_id != current["id"]:
            raise ValueError("Response item_id does not match the current routed item.")

        record = self.router.answer_item(item_id, response)
        self.responses[item_id] = response
        self.path.append(
            {
                "step": self.answered_count,
                "item_id": item_id,
                "text": current["text"],
                "dimension": current["dimension"],
                "key": current["key"],
                "response": response,
                "theta_after": record["theta_after"],
            }
        )
        return {
            "session_id": self.session_id,
            "accepted": True,
            "progress": self.progress(),
            "complete": self.is_complete,
        }

    def result(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "disclaimer": DISCLAIMER,
            "scoring_model": self.scoring_model,
            "progress": self.progress(),
            "trait_estimates": self.router.trait_estimates(),
            "irt_t_scores": self.router.tendency_t_scores(),
            "classical_big5": self.classical.score(self.responses),
            "dimension_answer_counts": self.router.dimension_answer_counts(),
            "path": self.path,
        }
