from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProgressEstimateRecord:
    param_mode: str
    scoring_model: str
    coverage_min_per_dimension: int
    stop_mean_standard_error: float
    estimated_total_items: int
    confidence_profile: str


class ProgressEstimator:
    def __init__(self, table_path: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        self.table_path = Path(table_path) if table_path is not None else root / "data" / "progress_estimates.json"
        payload = json.loads(self.table_path.read_text(encoding="utf-8"))
        self.version = str(payload.get("version", "unknown"))
        self.records = [
            ProgressEstimateRecord(
                param_mode=str(record["param_mode"]),
                scoring_model=str(record["scoring_model"]),
                coverage_min_per_dimension=int(record["coverage_min_per_dimension"]),
                stop_mean_standard_error=float(record["stop_mean_standard_error"]),
                estimated_total_items=int(record["estimated_total_items"]),
                confidence_profile=str(record["confidence_profile"]),
            )
            for record in payload["records"]
        ]

    def estimate(
        self,
        *,
        param_mode: str,
        scoring_model: str,
        coverage_min_per_dimension: int,
        stop_mean_standard_error: float,
        answered: int,
        max_items: int,
        complete: bool,
        min_items_met: bool,
        coverage_ready: bool,
        standard_error_ready: bool,
        stability_ready: bool,
        stopped_by: str,
        early_stop_candidate: bool,
        confirmation_items_remaining: int,
    ) -> dict[str, object]:
        record = self._match_record(
            param_mode=param_mode,
            scoring_model=scoring_model,
            coverage_min_per_dimension=coverage_min_per_dimension,
            stop_mean_standard_error=stop_mean_standard_error,
        )
        if record is None:
            estimated_total_items = max_items
            estimate_source = "fallback_max_items"
            confidence_profile = "custom configuration"
        else:
            estimated_total_items = max(record.estimated_total_items, 1)
            estimate_source = "lookup_table"
            confidence_profile = record.confidence_profile

        display_answered = answered if complete else min(answered + 1, estimated_total_items)
        estimated_completion_percent = int(round((display_answered / max(estimated_total_items, 1)) * 100))
        estimated_completion_percent = max(1 if not complete else 0, min(100, estimated_completion_percent))

        return {
            "estimated_total_items": estimated_total_items,
            "display_answered": display_answered,
            "estimated_completion_percent": estimated_completion_percent,
            "estimate_source": estimate_source,
            "confidence_profile": confidence_profile,
            "lookup_version": self.version,
            "evidence_stage": self._evidence_stage(
                min_items_met=min_items_met,
                coverage_ready=coverage_ready,
                standard_error_ready=standard_error_ready,
                stability_ready=stability_ready,
                stopped_by=stopped_by,
                early_stop_candidate=early_stop_candidate,
                confirmation_items_remaining=confirmation_items_remaining,
            ),
        }

    def _match_record(
        self,
        *,
        param_mode: str,
        scoring_model: str,
        coverage_min_per_dimension: int,
        stop_mean_standard_error: float,
    ) -> ProgressEstimateRecord | None:
        exact = [
            record
            for record in self.records
            if record.param_mode == param_mode
            and record.scoring_model == scoring_model
            and record.coverage_min_per_dimension == coverage_min_per_dimension
            and abs(record.stop_mean_standard_error - stop_mean_standard_error) < 1e-9
        ]
        if exact:
            return exact[0]

        nearby = [
            record
            for record in self.records
            if record.param_mode == param_mode
            and record.scoring_model == scoring_model
            and record.coverage_min_per_dimension == coverage_min_per_dimension
        ]
        if nearby:
            return min(nearby, key=lambda record: abs(record.stop_mean_standard_error - stop_mean_standard_error))
        return None

    @staticmethod
    def _evidence_stage(
        *,
        min_items_met: bool,
        coverage_ready: bool,
        standard_error_ready: bool,
        stability_ready: bool,
        stopped_by: str,
        early_stop_candidate: bool,
        confirmation_items_remaining: int,
    ) -> str:
        if stopped_by == "max_items_cap":
            return "item cap reached"
        if stopped_by == "item_bank_exhausted":
            return "item bank exhausted"
        if stopped_by in {"screening_confirmed", "stability_threshold"}:
            return "confidence target reached"
        if stopped_by == "screening_plateau":
            return "screening plateau reached"
        if stopped_by == "confirmation_window" or confirmation_items_remaining > 0:
            return "confirmation window"
        if stopped_by == "screening_candidate" or early_stop_candidate:
            return "early stop candidate"
        if not min_items_met:
            return "building minimum evidence"
        if not coverage_ready:
            return "coverage in progress"
        if stopped_by == "screening_gate":
            return "early confidence screening"
        if not standard_error_ready:
            return "confidence refining"
        if not stability_ready:
            return "stability refining"
        return "confidence target reached"
