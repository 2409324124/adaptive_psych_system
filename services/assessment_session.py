from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from engine import AdaptiveMMPIRouter, ClassicalBigFiveScorer, resolve_param_source
from services.progress_estimator import ProgressEstimator
from services.result_interpreter import ResultInterpreter
from services.stability_analyzer import StabilityAnalyzer


DISCLAIMER = "\u672c\u7cfb\u7edf\u4ec5\u4f5c\u4e3a\u5fc3\u7406\u7279\u8d28\u7b5b\u67e5\u4e0e\u8f85\u52a9\u53c2\u8003\u5de5\u5177\uff0c\u7edd\u5bf9\u4e0d\u53ef\u66ff\u4ee3\u4e13\u4e1a\u7cbe\u795e\u79d1\u4e34\u5e8a\u8bca\u65ad\u3002"
SCREENING_STOP_MEAN_STANDARD_ERROR = 0.85
REFINEMENT_ITEM_TRIGGER = 15
STABILITY_SE_RELAXATION = 0.1


@dataclass
class AssessmentSession:
    scoring_model: str = "binary_2pl"
    max_items: int = 30
    min_items: int = 5
    device: str | None = None
    param_mode: str | None = "keyed"
    param_path: str | None = None
    coverage_min_per_dimension: int = 2
    stop_mean_standard_error: float = 0.65
    stop_stability_score: float = 0.7
    session_id: str = field(default_factory=lambda: str(uuid4()))
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
        self.progress_estimator = ProgressEstimator()
        self.interpreter = ResultInterpreter()
        self.stability_analyzer = StabilityAnalyzer()
        self.responses: dict[str, int] = {}
        self.path: list[dict[str, object]] = []
        self.active_item: dict[str, object] | None = None
        self.user_comments: list[str] = []
        self.comment_submitted: bool = False

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

    def stability(self) -> dict[str, object]:
        return self.stability_analyzer.evaluate(
            history=self.router.history,
            path=self.path,
            dimensions=self.router.dimensions,
            stop_threshold=self.stop_stability_score,
        )

    def _progress_state(self) -> dict[str, int | float | bool]:
        answered = self.answered_count
        uncertainty = self.router.uncertainty_summary()
        counts = self.router.dimension_answer_counts()
        stability = self.stability()
        min_items_met = answered >= self.min_items
        coverage_ready = min(counts.values()) >= self.coverage_min_per_dimension
        stability_ready = bool(stability["stability_ready"])
        screening_stop_mean_standard_error = max(self.stop_mean_standard_error, SCREENING_STOP_MEAN_STANDARD_ERROR)
        refined_stop_mean_standard_error = self.stop_mean_standard_error + (
            STABILITY_SE_RELAXATION if stability_ready else 0.0
        )
        screening_threshold_met = uncertainty["mean_standard_error"] <= screening_stop_mean_standard_error
        refinement_active = answered > REFINEMENT_ITEM_TRIGGER and screening_threshold_met
        effective_stop_mean_standard_error = (
            refined_stop_mean_standard_error if refinement_active else screening_stop_mean_standard_error
        )
        se_threshold_met = uncertainty["mean_standard_error"] <= effective_stop_mean_standard_error

        if answered >= self.max_items:
            complete = True
            stopped_by = "max_items_cap"
            precision_mode = "item_cap"
        elif self.router.remaining_count <= 0:
            complete = True
            stopped_by = "item_bank_exhausted"
            precision_mode = "item_bank_exhausted"
        elif not min_items_met:
            complete = False
            stopped_by = "min_items_gate"
            precision_mode = "minimum_evidence"
        elif not coverage_ready:
            complete = False
            stopped_by = "coverage_gate"
            precision_mode = "coverage"
        elif answered > REFINEMENT_ITEM_TRIGGER and not screening_threshold_met:
            complete = True
            stopped_by = "screening_plateau"
            precision_mode = "screening_plateau"
        elif not screening_threshold_met:
            complete = False
            stopped_by = "screening_gate"
            precision_mode = "screening"
        elif answered <= REFINEMENT_ITEM_TRIGGER:
            if stability_ready:
                complete = True
                stopped_by = "screening_threshold"
            else:
                complete = False
                stopped_by = "stability_gate"
            precision_mode = "screening"
        elif not se_threshold_met:
            complete = False
            stopped_by = "standard_error_gate"
            precision_mode = "refining"
        elif not stability_ready:
            complete = False
            stopped_by = "stability_gate"
            precision_mode = "refining"
        else:
            complete = True
            stopped_by = "stability_threshold"
            precision_mode = "refining"

        return {
            "answered": answered,
            "min_items": self.min_items,
            "max_items": self.max_items,
            "remaining": max(0, self.max_items - answered),
            "mean_standard_error": uncertainty["mean_standard_error"],
            "confidence_ready": se_threshold_met,
            "coverage_min_per_dimension": self.coverage_min_per_dimension,
            "stop_mean_standard_error": self.stop_mean_standard_error,
            "screening_stop_mean_standard_error": screening_stop_mean_standard_error,
            "refinement_item_trigger": REFINEMENT_ITEM_TRIGGER,
            "screening_threshold_ready": screening_threshold_met,
            "effective_stop_mean_standard_error": effective_stop_mean_standard_error,
            "stop_stability_score": self.stop_stability_score,
            "min_items_met": min_items_met,
            "coverage_ready": coverage_ready,
            "standard_error_ready": se_threshold_met,
            "stability_ready": stability_ready,
            "stability_score": float(stability["stability_score"]),
            "stability_stage": str(stability["stability_stage"]),
            "precision_mode": precision_mode,
            "stopped_by": stopped_by,
            "complete": complete,
        }

    def progress(self) -> dict[str, int | float | bool | str]:
        return self._progress_state()

    def progress_estimate(self) -> dict[str, object]:
        progress = self.progress()
        return self.progress_estimator.estimate(
            param_mode=str(self.param_mode),
            scoring_model=self.scoring_model,
            coverage_min_per_dimension=self.coverage_min_per_dimension,
            stop_mean_standard_error=self.stop_mean_standard_error,
            answered=int(progress["answered"]),
            max_items=self.max_items,
            complete=bool(progress["complete"]),
            min_items_met=bool(progress["min_items_met"]),
            coverage_ready=bool(progress["coverage_ready"]),
            standard_error_ready=bool(progress["standard_error_ready"]),
            stability_ready=bool(progress["stability_ready"]),
            stopped_by=str(progress["stopped_by"]),
        )

    def next_question(self) -> dict[str, object] | None:
        self.touch()
        if self.is_complete:
            self.active_item = None
            return None
        if self.active_item is not None:
            self.active_item["progress"] = self.progress()
            self.active_item["progress_estimate"] = self.progress_estimate()
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
            "progress_estimate": self.progress_estimate(),
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
            "progress_estimate": self.progress_estimate(),
            "complete": self.is_complete,
        }

    def add_comment(self, comment: str) -> dict[str, object]:
        self.touch()
        text = comment.strip()
        if not text:
            raise ValueError("Comment must not be empty.")
        self.user_comments.append(text)
        self.comment_submitted = True
        return {
            "session_id": self.session_id,
            "accepted": True,
            "comment_submitted": self.comment_submitted,
            "user_comments": list(self.user_comments),
        }

    def restart(self) -> None:
        self.router.reset()
        self.responses.clear()
        self.path.clear()
        self.active_item = None
        self.user_comments.clear()
        self.comment_submitted = False
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
            "stop_stability_score": self.stop_stability_score,
            "device": self.device,
            "comment_submitted": self.comment_submitted,
            "user_comments": list(self.user_comments),
            **self.parameter_summary(),
            "progress": self.progress(),
            "progress_estimate": self.progress_estimate(),
            "stability": self.stability(),
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
        stability = self.stability()
        return {
            **self.summary(),
            "disclaimer": DISCLAIMER,
            "trait_estimates": self.router.trait_estimates(),
            "irt_t_scores": irt_t_scores,
            "standard_errors": standard_errors,
            "uncertainty": uncertainty,
            "stability": stability,
            "classical_big5": self.classical.score(self.responses),
            "dimension_answer_counts": dimension_answer_counts,
            "interpretation": self.interpreter.interpret(
                irt_t_scores=irt_t_scores,
                dimension_answer_counts=dimension_answer_counts,
            ),
            "responses": self.responses,
            "path": self.path,
            "cat_category": None,
            "cat_name": None,
            "cat_image": None,
            "cat_analysis": None,
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, object]) -> AssessmentSession:
        session = cls(
            scoring_model=str(payload["scoring_model"]),
            max_items=int(payload["max_items"]),
            min_items=int(payload.get("min_items", 5)),
            device=payload.get("device"),
            param_mode=payload.get("param_mode", "keyed"),
            param_path=payload.get("param_path"),
            coverage_min_per_dimension=int(payload.get("coverage_min_per_dimension", 2)),
            stop_mean_standard_error=float(payload.get("stop_mean_standard_error", 0.65)),
            stop_stability_score=float(payload.get("stop_stability_score", 0.7)),
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
        session.user_comments = [str(comment) for comment in payload.get("user_comments", [])]
        session.comment_submitted = bool(payload.get("comment_submitted", bool(session.user_comments)))
        session.updated_at = str(payload.get("updated_at") or datetime.now(UTC).isoformat())
        return session
