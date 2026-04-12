from __future__ import annotations

from pathlib import Path


DEFAULT_PARAM_MODE = "keyed"
PARAM_MODE_FILES = {
    "legacy": "mock_params.pt",
    "keyed": "mock_params_keyed.pt",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def infer_param_mode(param_path: str | Path) -> str:
    name = Path(param_path).name
    for mode, filename in PARAM_MODE_FILES.items():
        if name == filename:
            return mode
    return "custom"


def resolve_param_source(
    *,
    param_mode: str | None = None,
    param_path: str | Path | None = None,
    root: Path | None = None,
) -> tuple[str, Path]:
    """Resolve the experiment-facing param mode to a concrete torch parameter file path.

    Routers intentionally stay agnostic about param modes and only consume the resolved path.
    Sessions, API handlers, and scripts should call this helper once and then pass both the
    human-readable mode and resolved file path downstream for traceability.
    """
    base_root = root or project_root()
    if param_path is not None:
        resolved_path = Path(param_path)
        if not resolved_path.is_absolute():
            resolved_path = base_root / resolved_path
        mode = param_mode or infer_param_mode(resolved_path)
        return mode, resolved_path.resolve()

    mode = param_mode or DEFAULT_PARAM_MODE
    if mode not in PARAM_MODE_FILES:
        raise ValueError(f"Unknown param_mode: {mode}")
    return mode, (base_root / "data" / PARAM_MODE_FILES[mode]).resolve()
