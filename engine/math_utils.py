from __future__ import annotations

from typing import Literal

import torch


NeutralPolicy = Literal["skip", "zero"]
ResponseSource = Literal["binary", "likert", "llm"]


def resolve_device(preferred: str | torch.device | None = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def mirt_2pl_probability(theta: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    theta = theta.to(device=a.device, dtype=a.dtype)
    b = b.to(device=a.device, dtype=a.dtype)
    logits = a @ theta - b
    return sigmoid(logits)


def binary_fisher_information(theta: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    probabilities = mirt_2pl_probability(theta, a, b)
    discrimination_power = torch.sum(a * a, dim=-1)
    return probabilities * (1.0 - probabilities) * discrimination_power


def binary_fisher_information_matrix(
    theta: torch.Tensor,
    item_a: torch.Tensor,
    item_b: torch.Tensor,
) -> torch.Tensor:
    probability = mirt_2pl_probability(theta, item_a.unsqueeze(0), item_b.unsqueeze(0))[0]
    return probability * (1.0 - probability) * torch.outer(item_a, item_a)


def likert_to_binary(response: int, *, neutral_policy: NeutralPolicy = "skip") -> float | None:
    if response not in {1, 2, 3, 4, 5}:
        raise ValueError("Likert response must be an integer from 1 to 5.")
    if response <= 2:
        return 0.0
    if response >= 4:
        return 1.0
    if neutral_policy == "skip":
        return None
    return 0.5


def response_to_target(
    response: int | float,
    *,
    source: ResponseSource = "likert",
    neutral_policy: NeutralPolicy = "skip",
) -> float | None:
    if source == "likert":
        if not isinstance(response, int):
            raise ValueError("Likert response must be an integer from 1 to 5.")
        return likert_to_binary(response, neutral_policy=neutral_policy)

    if source == "binary":
        target = float(response)
        if target not in {0.0, 1.0}:
            raise ValueError("Binary response must be 0 or 1.")
        return target

    target = float(response)
    if not 0.0 <= target <= 1.0:
        raise ValueError("LLM response weight must be between 0.0 and 1.0.")
    return target


def binary_theta_update(
    theta: torch.Tensor,
    item_a: torch.Tensor,
    item_b: torch.Tensor,
    response: int | float,
    *,
    source: ResponseSource = "likert",
    response_weight: float = 1.0,
    learning_rate: float = 0.35,
    neutral_policy: NeutralPolicy = "skip",
    max_abs_theta: float = 4.0,
) -> torch.Tensor:
    if response_weight <= 0.0:
        raise ValueError("response_weight must be positive.")

    target = response_to_target(response, source=source, neutral_policy=neutral_policy)
    if target is None:
        return theta.clone()

    probability = mirt_2pl_probability(theta, item_a.unsqueeze(0), item_b.unsqueeze(0))[0]
    gradient = (torch.as_tensor(target, device=theta.device, dtype=theta.dtype) - probability) * item_a
    updated = theta + (learning_rate * response_weight) * gradient
    return torch.clamp(updated, min=-max_abs_theta, max=max_abs_theta)


def grm_thresholds_from_location(
    b: torch.Tensor,
    *,
    n_categories: int = 5,
    spacing: tuple[float, ...] = (-1.2, -0.4, 0.4, 1.2),
) -> torch.Tensor:
    if n_categories != 5:
        raise ValueError("The MVP GRM helper currently supports exactly 5 Likert categories.")
    offsets = torch.as_tensor(spacing, device=b.device, dtype=b.dtype)
    return b.unsqueeze(-1) + offsets


def grm_category_probabilities(theta: torch.Tensor, a: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    theta = theta.to(device=a.device, dtype=a.dtype)
    logits = a @ theta
    cumulative = sigmoid(logits.unsqueeze(-1) - thresholds)
    first = 1.0 - cumulative[..., :1]
    middle = cumulative[..., :-1] - cumulative[..., 1:]
    last = cumulative[..., -1:]
    probabilities = torch.cat([first, middle, last], dim=-1)
    return torch.clamp(probabilities, min=1e-7, max=1.0)


def grm_fisher_information(theta: torch.Tensor, a: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    probabilities = grm_category_probabilities(theta, a, thresholds)
    scores = torch.arange(1, probabilities.shape[-1] + 1, device=a.device, dtype=a.dtype)
    expected = torch.sum(probabilities * scores, dim=-1)
    variance = torch.sum(probabilities * (scores - expected.unsqueeze(-1)) ** 2, dim=-1)
    discrimination_power = torch.sum(a * a, dim=-1)
    return variance * discrimination_power


def grm_fisher_information_matrix(
    theta: torch.Tensor,
    item_a: torch.Tensor,
    item_thresholds: torch.Tensor,
) -> torch.Tensor:
    probabilities = grm_category_probabilities(
        theta,
        item_a.unsqueeze(0),
        item_thresholds.unsqueeze(0),
    )[0]
    scores = torch.arange(1, probabilities.shape[-1] + 1, device=item_a.device, dtype=item_a.dtype)
    expected = torch.sum(probabilities * scores)
    variance = torch.sum(probabilities * (scores - expected) ** 2)
    return variance * torch.outer(item_a, item_a)


def grm_theta_update(
    theta: torch.Tensor,
    item_a: torch.Tensor,
    item_thresholds: torch.Tensor,
    response: int,
    *,
    learning_rate: float = 0.08,
    max_abs_theta: float = 4.0,
) -> torch.Tensor:
    if response not in {1, 2, 3, 4, 5}:
        raise ValueError("Likert response must be an integer from 1 to 5.")

    working_theta = theta.detach().clone().requires_grad_(True)
    probabilities = grm_category_probabilities(
        working_theta,
        item_a.unsqueeze(0),
        item_thresholds.unsqueeze(0),
    )[0]
    log_likelihood = torch.log(probabilities[response - 1])
    log_likelihood.backward()
    gradient = working_theta.grad
    if gradient is None:
        return theta.clone()
    updated = theta + learning_rate * gradient.detach()
    return torch.clamp(updated, min=-max_abs_theta, max=max_abs_theta)
