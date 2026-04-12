from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from engine import AdaptiveMMPIRouter, ClassicalBigFiveScorer, resolve_param_source
from services.result_interpreter import ResultInterpreter


DISCLAIMER = "\u672c\u7cfb\u7edf\u4ec5\u4f5c\u4e3a\u5fc3\u7406\u7279\u8d28\u7b5b\u67e5\u4e0e\u8f85\u52a9\u53c2\u8003\u5de5\u5177\uff0c\u7edd\u5bf9\u4e0d\u53ef\u66ff\u4ee3\u4e13\u4e1a\u7cbe\u795e\u79d1\u4e34\u5e8a\u8bca\u65ad\u3002"


@dataclass
class AssessmentSession:
    scoring_model: str = "binary_2pl"
    max_items: int = 12
    min_items: int = 8
    device: str | None = None
    param_mode: str | None = None
    param_path: str | None = None
    coverage_min_per_dimension: int = 2
    stop_mean_standard_error: float = 0.85
    session_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        resolved_mode, resolved_path = resolve_param_source(param_mode=self.param_mode, param_path=self.param_path)
        self.param_mode = resolved_mode
        self.param_path = str(resolved_path)
        self.router = AdaptiveMMPIRouter(
            scoring_model=self.scoring_model,
            device=self.device,
            param_path=resolved_path,
            coverage_min_per_dimension=self.coverage_min_per_dimension,
        )
        self.classical = ClassicalBigFiveScorer(item_path=self.router.item_path)
        self.interpreter = ResultInterpreter()
        self.responses: dict[str, int] = {}
        self.path: list[dict[str, object]] = []
        self.active_item: dict[str, object] | None = None

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC).isoformat()

    @property
    def answered_count(self) -> int:
        return self.router.answered_count

    @property
    def is_complete(self) -> bool:
        return bool(self._progress_state()["complete"])

    def parameter_summary(self) -> dict[str, object]:
        return {
            "param_mode": self.param_mode,
            "param_path": self.param_path,
            "key_aligned": self.router.key_aligned,
            "param_metadata": dict(self.router.param_metadata),
        }

    def _progress_state(self) -> dict[str, int | float | bool]:
        answered = self.answered_count
        uncertainty = self.router.uncertainty_summary()
        counts = self.router.dimension_answer_counts()
        min_items_met = answered >= self.min_items
        coverage_ready = min(counts.values()) >= self.coverage_min_per_dimension
        se_threshold_met = uncertainty["mean_standard_error"] <= self.stop_mean_standard_error

        if answered >= self.max_items:
            complete = True
            stopped_by = "max_items_cap"
        elif self.router.remaining_count <= 0:
            complete = True
            stopped_by = "item_bank_exhausted"
        elif not min_items_met:
            complete = False
            stopped_by = "min_items_gate"
        elif not coverage_ready:
            complete = False
            stopped_by = "coverage_gate"
        elif se_threshold_met:
            complete = True
            stopped_by = "standard_error_threshold"
        else:
            complete = False
            stopped_by = "standard_error_gate"

        return {
            "answered": answered,
            "min_items": self.min_items,
            "max_items": self.max_items,
            "remaining": max(0, self.max_items - answered),
            "mean_standard_error": uncertainty["mean_standard_error"],
            "confidence_ready": uncertainty["confidence_ready"],
            "coverage_min_per_dimension": self.coverage_min_per_dimension,
            "stop_mean_standard_error": self.stop_mean_standard_error,
            "min_items_met": min_items_met,
            "coverage_ready": coverage_ready,
            "standard_error_ready": se_threshold_met,
            "stopped_by": stopped_by,
            "complete": complete,
        }

    def progress(self) -> dict[str, int | float | bool | str]:
        return self._progress_state()

    def next_question(self) -> dict[str, object] | None:
        self.touch()
        if self.is_complete:
            self.active_item = None
            return None
        if self.active_item is not None:
            self.active_item["progress"] = self.progress()
            return self.active_item
        item = self.router.select_next_item()
        if item is None:
            self.active_item = None
            return None
        self.active_item = {
            "session_id": self.session_id,
            "item_id": item["id"],
            "text": item["text"],
            "dimension": item["dimension"],
            "key": item["key"],
            "response_scale": item["response_scale"],
            "progress": self.progress(),
        }
        return self.active_item

    def submit_response(self, item_id: str, response: int) -> dict[str, object]:
        self.touch()
        if self.is_complete:
            raise ValueError("This assessment session is already complete.")
        current = self.active_item or self.next_question()
        if current is None:
            raise ValueError("No active item is available.")
        if item_id != current["item_id"]:
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
                "keyed_response": record["keyed_response"],
                "theta_after": record["theta_after"],
            }
        )
        self.active_item = None
        return {
            "session_id": self.session_id,
            "accepted": True,
            "progress": self.progress(),
            "complete": self.is_complete,
        }

    def restart(self) -> None:
        self.router.reset()
        self.responses.clear()
        self.path.clear()
        self.active_item = None
        self.touch()

    def summary(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "scoring_model": self.scoring_model,
            "max_items": self.max_items,
            "min_items": self.min_items,
            "coverage_min_per_dimension": self.coverage_min_per_dimension,
            "stop_mean_standard_error": self.stop_mean_standard_error,
            "device": self.device,
            **self.parameter_summary(),
            "progress": self.progress(),
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.summary(),
            "responses": self.responses,
            "path": self.path,
            "active_item": self.active_item,
        }

    def result(self) -> dict[str, object]:
        irt_t_scores = self.router.tendency_t_scores()
        dimension_answer_counts = self.router.dimension_answer_counts()
        standard_errors = self.router.standard_errors()
        uncertainty = self.router.uncertainty_summary()
        return {
            **self.summary(),
            "disclaimer": DISCLAIMER,
            "trait_estimates": self.router.trait_estimates(),
            "irt_t_scores": irt_t_scores,
            "standard_errors": standard_errors,
            "uncertainty": uncertainty,
            "classical_big5": self.classical.score(self.responses),
            "dimension_answer_counts": dimension_answer_counts,
            "interpretation": self.interpreter.interpret(
                irt_t_scores=irt_t_scores,
                dimension_answer_counts=dimension_answer_counts,
            ),
            "path": self.path,
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, object]) -> AssessmentSession:
        session = cls(
            scoring_model=str(payload["scoring_model"]),
            max_items=int(payload["max_items"]),
            min_items=int(payload.get("min_items", 8)),
            device=payload.get("device"),
            param_mode=payload.get("param_mode"),
            param_path=payload.get("param_path"),
            coverage_min_per_dimension=int(payload.get("coverage_min_per_dimension", 2)),
            stop_mean_standard_error=float(payload.get("stop_mean_standard_error", 0.85)),
            session_id=str(payload["session_id"]),
            created_at=str(payload.get("created_at") or datetime.now(UTC).isoformat()),
            updated_at=str(payload.get("updated_at") or datetime.now(UTC).isoformat()),
        )
        for step in payload.get("path", []):
            item_id = str(step["item_id"])
            response = int(step["response"])
            record = session.router.answer_item(item_id, response)
            session.responses[item_id] = response
            session.path.append(
                {
                    "step": int(step.get("step", len(session.path) + 1)),
                    "item_id": item_id,
                    "text": step["text"],
                    "dimension": step["dimension"],
                    "key": int(step["key"]),
                    "response": response,
                    "keyed_response": record["keyed_response"],
                    "theta_after": record["theta_after"],
                }
            )
        active_item = payload.get("active_item")
        session.active_item = dict(active_item) if isinstance(active_item, dict) else None
        session.updated_at = str(payload.get("updated_at") or datetime.now(UTC).isoformat())
        return session
