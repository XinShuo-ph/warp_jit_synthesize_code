"""Warp device helpers shared across scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedDevice:
    """Normalized device description for Warp compilation/launch."""

    name: str  # "cpu" | "cuda"


def resolve_warp_device(requested: str) -> ResolvedDevice:
    """
    Resolve and validate a Warp device name.

    Rules:
    - CPU is always allowed.
    - CUDA must be available; otherwise this raises with a clear message.
    """
    import warp as wp

    if requested is None:
        requested = "cpu"

    device = requested.strip().lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{requested}'. Expected 'cpu' or 'cuda'.")

    # For explicit CUDA requests, fail fast if unavailable.
    # Prefer `is_cuda_available()` since `is_device_available("cuda")` may touch the
    # CUDA runtime in ways that raise in CPU-only environments.
    if device == "cuda" and not wp.is_cuda_available():
        raise RuntimeError(
            "CUDA device requested but not available in this environment. "
            "On a CUDA machine, ensure Warp detects CUDA and the CUDA toolkit is installed."
        )

    return ResolvedDevice(name=device)

