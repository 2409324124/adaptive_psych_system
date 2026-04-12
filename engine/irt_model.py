from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .math_utils import (
    binary_fisher_information,
    binary_fisher_information_matrix,
    binary_theta_update,
    grm_fisher_information,
    grm_fisher_information_matrix,
    grm_theta_update,
    grm_thresholds_from_location,
    resolve_device,
)


ScoringModel = Literal["binary_2pl", "grm"]


@dataclass(frozen=True)
class AdaptiveItem:
    index: int
    item_id: str
    text: str
    dimension: str
    key: int

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "id": self.item_id,
            "text": self.text,
            "dimension": self.dimension,
            "key": self.key,
        }


class AdaptiveMMPIRouter:
    def __init__(
        self,
        *,
        item_path: str | Path | None = None,
        param_path: str | Path | None = None,
        scoring_model: ScoringModel = "binary_2pl",
        device: str | torch.device | None = None,
        binary_learning_rate: float = 0.35,
        grm_learning_rate: float = 0.08,
        coverage_min_per_dimension: int = 2,
    ) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.item_path = Path(item_path) if item_path is not None else self.root / "data" / "ipip_items.json"
        self.param_path = Path(param_path) if param_path is not None else self.root / "data" / "mock_params.pt"
        self.scoring_model = scoring_model
        self.device = resolve_device(device)
        self.binary_learning_rate = binary_learning_rate
        self.grm_learning_rate = grm_learning_rate
        self.coverage_min_per_dimension = coverage_min_per_dimension

        self.items, self.dimensions, self.response_scale = self._load_items()
        self.a, self.b, self.param_metadata = self._load_params()
        self.thresholds = grm_thresholds_from_location(self.b)
        self.theta = torch.zeros(len(self.dimensions), device=self.device, dtype=self.a.dtype)
        self.information_matrix = torch.zeros(
            (len(self.dimensions), len(self.dimensions)),
            device=self.device,
            dtype=self.a.dtype,
        )
        self.answered_indices: set[int] = set()
        self.history: list[dict[str, object]] = []

        self._validate_shapes()

    def _load_items(self) -> tuple[list[AdaptiveItem], list[str], dict[str, object]]:
        with self.item_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        items = [
            AdaptiveItem(
                index=index,
                item_id=item["id"],
                text=item["text"],
                dimension=item["dimension"],
                key=int(item.get("key", 1)),
            )
            for index, item in enumerate(payload["items"])
        ]
        return items, list(payload["dimensions"]), dict(payload["response_scale"])

    def _load_params(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        payload = torch.load(self.param_path, map_location=self.device, weights_only=True)
        a = payload["a"].to(self.device, dtype=torch.float32)
        b = payload["b"].to(self.device, dtype=torch.float32)
        metadata = dict(payload.get("metadata", {}))
        return a, b, metadata

    def _validate_shapes(self) -> None:
        if self.a.ndim != 2:
            raise ValueError("Parameter tensor 'a' must have shape (n_items, n_dimensions).")
        if self.b.ndim != 1:
            raise ValueError("Parameter tensor 'b' must have shape (n_items,).")
        if self.a.shape[0] != len(self.items) or self.b.shape[0] != len(self.items):
            raise ValueError("Item metadata and parameter tensors must have the same item count.")
        if self.a.shape[1] != len(self.dimensions):
            raise ValueError("Parameter tensor dimension count must match item dimensions.")
        if self.scoring_model not in {"binary_2pl", "grm"}:
            raise ValueError("scoring_model must be either 'binary_2pl' or 'grm'.")
        if self.coverage_min_per_dimension < 0:
            raise ValueError("coverage_min_per_dimension must be non-negative.")

    @property
    def key_aligned(self) -> bool:
        return bool(self.param_metadata.get("key_aligned", False))

    @property
    def answered_count(self) -> int:
        return len(self.answered_indices)

    @property
    def remaining_count(self) -> int:
        return len(self.items) - len(self.answered_indices)

    def reset(self) -> None:
        self.theta = torch.zeros(len(self.dimensions), device=self.device, dtype=self.a.dtype)
        self.information_matrix = torch.zeros(
            (len(self.dimensions), len(self.dimensions)),
            device=self.device,
            dtype=self.a.dtype,
        )
        self.answered_indices.clear()
        self.history.clear()

    def information_scores(self) -> torch.Tensor:
        if self.scoring_model == "binary_2pl":
            scores = binary_fisher_information(self.theta, self.a, self.b)
        else:
            scores = grm_fisher_information(self.theta, self.a, self.thresholds)

        if self.answered_indices:
            answered = torch.as_tensor(list(self.answered_indices), device=self.device, dtype=torch.long)
            scores = scores.clone()
            scores[answered] = -torch.inf
        return scores

    def fisher_information_matrix(self, item_id: str) -> torch.Tensor:
        index = self._index_for_item_id(item_id)
        if self.scoring_model == "binary_2pl":
            return binary_fisher_information_matrix(self.theta, self.a[index], self.b[index])
        return grm_fisher_information_matrix(self.theta, self.a[index], self.thresholds[index])

    def cumulative_information_matrix(self) -> torch.Tensor:
        return self.information_matrix.clone()

    def covariance_matrix(self, *, ridge: float = 1e-3) -> torch.Tensor:
        identity = torch.eye(len(self.dimensions), device=self.device, dtype=self.a.dtype)
        regularized = self.information_matrix + ridge * identity
        return torch.linalg.pinv(regularized)

    def standard_errors(self, *, ridge: float = 1e-3) -> dict[str, float]:
        covariance = self.covariance_matrix(ridge=ridge).detach().cpu()
        diagonal = torch.diagonal(covariance).clamp_min(0.0)
        return {
            dimension: float(torch.sqrt(diagonal[index]))
            for index, dimension in enumerate(self.dimensions)
        }

    def uncertainty_summary(self, *, ridge: float = 1e-3) -> dict[str, float | bool]:
        covariance = self.covariance_matrix(ridge=ridge).detach().cpu()
        diagonal = torch.diagonal(covariance).clamp_min(0.0)
        std_errors = torch.sqrt(diagonal)
        mean_se = float(std_errors.mean())
        max_se = float(std_errors.max())
        return {
            "mean_standard_error": mean_se,
            "max_standard_error": max_se,
            "confidence_ready": bool(mean_se <= 0.85 and self.answered_count >= len(self.dimensions)),
        }

    def select_next_item(self) -> dict[str, object] | None:
        if self.remaining_count <= 0:
            return None
        scores = self.information_scores()
        index = self._coverage_aware_index(scores)
        item = self.items[index].to_dict()
        item["response_scale"] = self.response_scale
        item["scoring_model"] = self.scoring_model
        return item

    def update_theta(
        self,
        item_id: str,
        response: int | float,
        *,
        source: Literal["likert", "binary", "llm"] = "likert",
        response_weight: float = 1.0,
    ) -> dict[str, object]:
        index = self._index_for_item_id(item_id)
        if index in self.answered_indices:
            raise ValueError(f"Item already answered: {item_id}")

        previous_theta = self.theta.clone()
        item_information = self.fisher_information_matrix(item_id)
        keyed_response = self._keyed_response(index, response, source=source)
        if self.scoring_model == "binary_2pl":
            self.theta = binary_theta_update(
                self.theta,
                self.a[index],
                self.b[index],
                keyed_response,
                source=source,
                response_weight=response_weight,
                learning_rate=self.binary_learning_rate,
            )
        else:
            if source != "likert":
                raise ValueError("GRM updates currently require Likert responses.")
            self.theta = grm_theta_update(
                self.theta,
                self.a[index],
                self.thresholds[index],
                int(keyed_response),
                learning_rate=self.grm_learning_rate,
            )

        self.answered_indices.add(index)
        self.information_matrix = self.information_matrix + item_information
        record = {
            "item_id": item_id,
            "response": response,
            "keyed_response": keyed_response,
            "response_source": source,
            "response_weight": response_weight,
            "scoring_model": self.scoring_model,
            "theta_before": previous_theta.detach().cpu().tolist(),
            "theta_after": self.theta.detach().cpu().tolist(),
            "information_trace_after": float(torch.trace(self.information_matrix).detach().cpu()),
        }
        self.history.append(record)
        return record

    def answer_item(self, item_id: str, response: int) -> dict[str, object]:
        return self.update_theta(item_id, response, source="likert", response_weight=1.0)

    def trait_estimates(self) -> dict[str, float]:
        theta_cpu = self.theta.detach().cpu()
        return {dimension: float(theta_cpu[index]) for index, dimension in enumerate(self.dimensions)}

    def tendency_t_scores(self) -> dict[str, float]:
        theta_cpu = self.theta.detach().cpu()
        return {
            dimension: float(50.0 + 10.0 * theta_cpu[index])
            for index, dimension in enumerate(self.dimensions)
        }

    def dimension_answer_counts(self) -> dict[str, int]:
        counts = {dimension: 0 for dimension in self.dimensions}
        for index in self.answered_indices:
            counts[self.items[index].dimension] += 1
        return counts

    def _coverage_aware_index(self, scores: torch.Tensor) -> int:
        if self.coverage_min_per_dimension <= 0:
            return int(torch.argmax(scores).item())

        counts = self.dimension_answer_counts()
        undercovered = {
            dimension
            for dimension, count in counts.items()
            if count < self.coverage_min_per_dimension
        }
        if not undercovered:
            return int(torch.argmax(scores).item())

        masked_scores = scores.clone()
        for item in self.items:
            if item.dimension not in undercovered:
                masked_scores[item.index] = -torch.inf

        if torch.isneginf(masked_scores).all():
            return int(torch.argmax(scores).item())
        return int(torch.argmax(masked_scores).item())

    def _keyed_response(
        self,
        index: int,
        response: int | float,
        *,
        source: Literal["likert", "binary", "llm"],
    ) -> int | float:
        if self.key_aligned:
            return response
        key = self.items[index].key
        if key >= 0:
            return response
        if source == "likert":
            return 6 - int(response)
        return 1.0 - float(response)

    def _index_for_item_id(self, item_id: str) -> int:
        for item in self.items:
            if item.item_id == item_id:
                return item.index
        raise KeyError(f"Unknown item_id: {item_id}")
