from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate key-aware mock IRT parameter tensors.")
    parser.add_argument(
        "--items",
        type=Path,
        default=Path("data/ipip_items.json"),
        help="Path to the item metadata JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mock_params_keyed.pt"),
        help="Output path for the generated torch parameter file.",
    )
    parser.add_argument("--seed", type=int, default=20260412, help="Random seed for reproducible mock parameters.")
    parser.add_argument(
        "--primary-loading",
        type=float,
        default=1.35,
        help="Base loading magnitude for the target dimension before adding noise.",
    )
    parser.add_argument(
        "--cross-loading-std",
        type=float,
        default=0.10,
        help="Standard deviation for cross-dimension loadings.",
    )
    parser.add_argument(
        "--primary-loading-std",
        type=float,
        default=0.12,
        help="Standard deviation added to the primary loading magnitude.",
    )
    parser.add_argument(
        "--difficulty-std",
        type=float,
        default=0.75,
        help="Standard deviation for mock difficulty parameters.",
    )
    return parser.parse_args()


def build_key_aware_params(
    *,
    items_path: Path,
    seed: int,
    primary_loading: float,
    primary_loading_std: float,
    cross_loading_std: float,
    difficulty_std: float,
) -> dict[str, object]:
    payload = json.loads(items_path.read_text(encoding="utf-8"))
    items = payload["items"]
    dimensions = list(payload["dimensions"])
    dim_index = {dimension: index for index, dimension in enumerate(dimensions)}

    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn((len(items), len(dimensions)), generator=generator, dtype=torch.float32) * cross_loading_std
    b = torch.randn(len(items), generator=generator, dtype=torch.float32) * difficulty_std

    for row_index, item in enumerate(items):
        target_index = dim_index[item["dimension"]]
        direction = 1.0 if int(item.get("key", 1)) >= 0 else -1.0
        magnitude_noise = torch.randn(1, generator=generator, dtype=torch.float32).item() * primary_loading_std
        loading = max(0.4, primary_loading + magnitude_noise)
        a[row_index, target_index] = direction * loading

    return {
        "a": a,
        "b": b,
        "metadata": {
            "generator": "generate_key_aware_mock_params.py",
            "seed": seed,
            "primary_loading": primary_loading,
            "primary_loading_std": primary_loading_std,
            "cross_loading_std": cross_loading_std,
            "difficulty_std": difficulty_std,
            "key_aligned": True,
            "dimensions": dimensions,
            "device": "cpu",
        },
    }


def main() -> None:
    args = parse_args()
    params = build_key_aware_params(
        items_path=args.items,
        seed=args.seed,
        primary_loading=args.primary_loading,
        primary_loading_std=args.primary_loading_std,
        cross_loading_std=args.cross_loading_std,
        difficulty_std=args.difficulty_std,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(params, args.output)
    print(f"Saved key-aware mock parameters to {args.output}")


if __name__ == "__main__":
    main()
