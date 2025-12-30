"""IR Extractor - Extracts JAX compiler IR (StableHLO/MLIR) from JITted functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence
import inspect

import jax
import jax.numpy as jnp


@dataclass
class ExtractedIR:
    """Container for extracted compiler IR from a JAX function."""

    kernel_name: str
    python_source: str
    cpu_code: str  # StableHLO/MLIR for CPU compilation target
    cuda_code: Optional[str] = None  # StableHLO/MLIR for GPU target (if available)
    cpu_backward_code: Optional[str] = None
    cuda_backward_code: Optional[str] = None


def _compiler_ir_text(lowered: Any) -> str:
    """
    Convert JAX lowered compiler IR to text.

    JAX has changed return types here a few times; handle the common shapes.
    """
    ir = lowered.compiler_ir(dialect="stablehlo")
    if hasattr(ir, "as_text"):
        return ir.as_text()
    return str(ir)


def _extract_for_device(
    fn: Callable[..., Any],
    example_args: Sequence[Any],
    device: jax.Device,
) -> str:
    jit_fn = jax.jit(fn, device=device)
    lowered = jit_fn.lower(*example_args)
    return _compiler_ir_text(lowered)


def _maybe_devices(kind: str) -> list[jax.Device]:
    try:
        return list(jax.devices(kind))
    except Exception:
        return []


def extract_ir(
    fn: Callable[..., Any],
    example_args: Sequence[Any],
    *,
    kernel_name: Optional[str] = None,
    enable_backward: bool = True,
    grad_argnums: tuple[int, ...] = (0,),
) -> ExtractedIR:
    """
    Extract compiler IR (StableHLO/MLIR text) for a JAX function.

    - CPU: always attempted.
    - CUDA: attempted only if a GPU device is present.
    - Backward: compiled as grad of sum(output) when possible.
    """
    name = kernel_name or getattr(fn, "__name__", "jax_fn")

    try:
        python_source = inspect.getsource(fn).strip()
    except Exception:
        python_source = f"# Source unavailable for {name}"

    cpu_devs = _maybe_devices("cpu")
    if not cpu_devs:
        raise RuntimeError("No CPU device found for JAX")

    cpu_code = _extract_for_device(fn, example_args, cpu_devs[0])

    cuda_code = None
    cuda_devs = _maybe_devices("gpu")
    if cuda_devs:
        try:
            cuda_code = _extract_for_device(fn, example_args, cuda_devs[0])
        except Exception:
            cuda_code = None

    cpu_backward_code = None
    cuda_backward_code = None

    if enable_backward:
        # Define a scalar loss to make grad well-defined.
        def loss(*args):
            out = fn(*args)
            return jnp.sum(out)

        try:
            bwd = jax.grad(loss, argnums=grad_argnums)
            cpu_backward_code = _extract_for_device(bwd, example_args, cpu_devs[0])
            if cuda_devs:
                try:
                    cuda_backward_code = _extract_for_device(bwd, example_args, cuda_devs[0])
                except Exception:
                    cuda_backward_code = None
        except Exception:
            # Not all functions/arg types are differentiable (e.g., integer ops).
            cpu_backward_code = None
            cuda_backward_code = None

    return ExtractedIR(
        kernel_name=name,
        python_source=python_source,
        cpu_code=cpu_code,
        cuda_code=cuda_code,
        cpu_backward_code=cpu_backward_code,
        cuda_backward_code=cuda_backward_code,
    )


def extract_ir_pair(
    fn: Callable[..., Any],
    example_args: Sequence[Any],
    *,
    device: str = "cpu",
) -> tuple[str, str]:
    """
    Extract Python→IR pair suitable for LLM training.

    Args:
        fn: JAX function (will be jitted)
        example_args: Example inputs used for lowering/compilation
        device: "cpu" or "cuda"
    """
    ir = extract_ir(fn, example_args)
    if device == "cuda" and ir.cuda_code:
        return (ir.python_source, ir.cuda_code)
    return (ir.python_source, ir.cpu_code)


if __name__ == "__main__":
    # Minimal smoke test
    def test_kernel(a: jnp.ndarray) -> jnp.ndarray:
        return a * 2.0

    args = (jnp.arange(8, dtype=jnp.float32),)
    ir = extract_ir(test_kernel, args)

    print("=== Kernel Name ===")
    print(ir.kernel_name)
    print("\n=== Python Source ===")
    print(ir.python_source)
    print("\n=== CPU StableHLO/MLIR (first 1500 chars) ===")
    print(ir.cpu_code[:1500])
    print("\n=== CUDA code available ===")
    print("Yes" if ir.cuda_code else "No")
