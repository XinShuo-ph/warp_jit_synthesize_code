"""IR Extractor - Extracts JAX compiler IR (HLO/StableHLO) from JAX functions.

This is the JAX analogue of the previous Warp C++/CUDA extractor. We treat:
- "cpp_code" as CPU HLO (forward + backward)
- "cuda_code" as GPU HLO (forward + backward), if a GPU backend is available
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import inspect

import jax
import jax.numpy as jnp


@dataclass
class ExtractedIR:
    """Container for extracted IR from a JAX function."""

    kernel_name: str
    python_source: str
    cpp_code: str  # CPU HLO (forward + backward)
    cuda_code: Optional[str] = None  # GPU HLO (forward + backward), if available


def _has_gpu_backend() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def _hlo_text(
    fun,
    args: tuple[Any, ...],
    backend: Optional[str],
    static_argnums: tuple[int, ...] = (),
) -> str:
    """Lower + extract textual HLO (or best-effort compiler IR)."""
    jit_kwargs = {}
    if backend is not None:
        jit_kwargs["backend"] = backend
    if static_argnums:
        jit_kwargs["static_argnums"] = static_argnums

    jit_fun = jax.jit(fun, **jit_kwargs)

    # Newer JAX: lowered.compiler_ir(dialect=...)
    try:
        lowered = jit_fun.lower(*args)
        for dialect in ("hlo", "stablehlo"):
            try:
                ir = lowered.compiler_ir(dialect=dialect)
                if hasattr(ir, "as_hlo_text"):
                    return ir.as_hlo_text()
                # Some versions return an MLIR module-like object
                return str(ir)
            except TypeError:
                # compiler_ir() without dialect (older/newer variants)
                break
            except Exception:
                continue

        ir = lowered.compiler_ir()
        if hasattr(ir, "as_hlo_text"):
            return ir.as_hlo_text()
        return str(ir)
    except Exception as e:
        raise RuntimeError(f"Failed to lower and extract compiler IR: {e}") from e


def extract_ir(
    kernel_fn,
    example_args: tuple[Any, ...],
    enable_backward: bool = True,
) -> ExtractedIR:
    """Extract CPU/GPU compiler IR for forward + backward."""
    try:
        python_source = inspect.getsource(kernel_fn)
    except OSError:
        python_source = repr(kernel_fn)

    def forward(*args):
        return kernel_fn(*args)

    def loss(*args):
        out = forward(*args)
        out_arr = jnp.asarray(out)
        return jnp.sum(out_arr)

    # Decide which args are differentiable (exclude ints/bools)
    grad_argnums = tuple(
        i for i, a in enumerate(example_args) if not isinstance(a, (bool, int))
    )
    # Make non-array control args static for lowering/autodiff (e.g. loop bounds)
    static_argnums = tuple(i for i, a in enumerate(example_args) if isinstance(a, (bool, int)))

    def backward(*args):
        if not enable_backward or len(grad_argnums) == 0:
            return ()
        g = jax.grad(loss, argnums=grad_argnums)(*args)
        return g

    cpu_forward = _hlo_text(forward, example_args, backend="cpu", static_argnums=static_argnums)
    cpu_backward = (
        _hlo_text(backward, example_args, backend="cpu", static_argnums=static_argnums)
        if enable_backward
        else ""
    )
    cpp_code = (
        "### FORWARD (CPU HLO)\n"
        + cpu_forward
        + "\n\n### BACKWARD (CPU HLO)\n"
        + cpu_backward
    )

    cuda_code = None
    if _has_gpu_backend():
        gpu_forward = _hlo_text(forward, example_args, backend="gpu", static_argnums=static_argnums)
        gpu_backward = (
            _hlo_text(backward, example_args, backend="gpu", static_argnums=static_argnums)
            if enable_backward
            else ""
        )
        cuda_code = (
            "### FORWARD (GPU HLO)\n"
            + gpu_forward
            + "\n\n### BACKWARD (GPU HLO)\n"
            + gpu_backward
        )

    return ExtractedIR(
        kernel_name=getattr(kernel_fn, "__name__", "kernel"),
        python_source=python_source,
        cpp_code=cpp_code,
        cuda_code=cuda_code,
    )


def extract_ir_pair(kernel_fn, example_args: tuple[Any, ...], device: str = "cpu") -> tuple[str, str]:
    """Extract a (python_source, ir_text) pair suitable for training."""
    ir = extract_ir(kernel_fn, example_args=example_args, enable_backward=True)
    if device == "cuda" and ir.cuda_code:
        return (ir.python_source, ir.cuda_code)
    return (ir.python_source, ir.cpp_code)


if __name__ == "__main__":
    # Basic smoke test
    def test_kernel(a):
        return a * 2.0

    args = (jnp.arange(8, dtype=jnp.float32),)
    ir = extract_ir(test_kernel, example_args=args)
    print("=== Kernel Name ===")
    print(ir.kernel_name)
    print("\n=== Python Source ===")
    print(ir.python_source)
    print("\n=== CPU IR (first 1500 chars) ===")
    print(ir.cpp_code[:1500])
    print("\n=== GPU IR available ===")
    print("Yes" if ir.cuda_code else "No")
