"""IR Extractor - Extracts compiler IR from JAX-jitted functions.

This repo originally extracted generated C++/CUDA from Warp kernels. The JAX
version extracts JAX compiler IR (e.g. StableHLO / HLO / LLVM dialect) for:
- the forward computation
- the backward computation (via `jax.grad` of a scalar loss derived from outputs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence
import inspect

import jax
import jax.numpy as jnp


@dataclass
class ExtractedIR:
    """Container for extracted IR from a JAX function."""

    kernel_name: str
    python_source: str
    cpp_code: str  # historical name; now JAX CPU compiler IR (forward + backward)
    cuda_code: Optional[str] = None  # historical name; now JAX GPU compiler IR (forward + backward)


def _compiler_ir_text(lowered: Any, dialect: str) -> str:
    """Best-effort wrapper across JAX versions for compiler IR text."""
    # JAX 0.4+ supports: lowered.compiler_ir(dialect=...)
    try:
        ir = lowered.compiler_ir(dialect=dialect)
    except TypeError:
        # Older signature: lowered.compiler_ir("stablehlo")
        ir = lowered.compiler_ir(dialect)

    # Some IR objects expose .as_text(), others stringify well.
    if hasattr(ir, "as_text"):
        return ir.as_text()
    return str(ir)


def _normalize_outputs_to_scalar(outputs: Any) -> jax.Array:
    """Convert arbitrary outputs to a scalar loss (for grad extraction)."""
    if isinstance(outputs, tuple):
        # Sum-of-sums over tuple leaves.
        return sum(_normalize_outputs_to_scalar(o) for o in outputs)
    if isinstance(outputs, (list,)):
        return sum(_normalize_outputs_to_scalar(o) for o in outputs)
    arr = jnp.asarray(outputs)
    if arr.shape == ():
        return arr
    return jnp.sum(arr)


def _differentiable_argnums(args: Sequence[Any]) -> tuple[int, ...]:
    """Select arg indices that are differentiable by JAX (inexact dtypes)."""
    argnums: list[int] = []
    for i, a in enumerate(args):
        # Skip Python scalars we treat as static.
        if isinstance(a, (int, bool, str)):
            continue
        try:
            arr = jnp.asarray(a)
        except Exception:
            continue
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            argnums.append(i)
    return tuple(argnums)


def _select_device(platform: str) -> Optional[jax.Device]:
    try:
        devs = jax.devices(platform)
    except Exception:
        return None
    return devs[0] if devs else None


def _lower_on_device(fn: Callable[..., Any], args: Sequence[Any], device: jax.Device):
    static_argnums = tuple(
        i
        for i, a in enumerate(args)
        # Treat Python scalars (loop bounds, flags, etc.) as static to avoid
        # dynamic control-flow limitations in reverse-mode AD.
        if isinstance(a, (int, bool, str))
    )
    # Prefer per-device jit if available.
    try:
        jitted = jax.jit(fn, device=device, static_argnums=static_argnums)
        return jitted.lower(*args)
    except TypeError:
        # Fall back to default_device context.
        try:
            with jax.default_device(device):
                jitted = jax.jit(fn, static_argnums=static_argnums)
                return jitted.lower(*args)
        except Exception:
            # Last resort: lower without forcing device.
            jitted = jax.jit(fn, static_argnums=static_argnums)
            return jitted.lower(*args)


def extract_ir(
    fn: Callable[..., Any],
    example_args: Sequence[Any],
    *,
    kernel_name: Optional[str] = None,
    python_source: Optional[str] = None,
    enable_backward: bool = True,
    dialect: str = "stablehlo",
    include_llvm: bool = False,
) -> ExtractedIR:
    """Extract JAX compiler IR for forward (+ backward).

    Notes:
    - `cpp_code`/`cuda_code` field names are kept for backward compatibility with
      the old Warp dataset format; they now contain JAX IR text.
    - Backward IR is extracted by compiling `jax.grad(loss_fn)` where `loss_fn`
      reduces the forward outputs to a scalar.
    """
    if kernel_name is None:
        kernel_name = getattr(fn, "__name__", "jax_fn")
    if python_source is None:
        try:
            python_source = inspect.getsource(fn)
        except Exception:
            python_source = f"# source unavailable for {kernel_name}"

    args = tuple(example_args)
    argnums = _differentiable_argnums(args)

    def loss_fn(*a):
        return _normalize_outputs_to_scalar(fn(*a))

    # CPU IR
    cpu_dev = _select_device("cpu")
    if cpu_dev is None:
        raise RuntimeError("No JAX CPU device available")

    fwd_lowered_cpu = _lower_on_device(fn, args, cpu_dev)
    fwd_cpu = _compiler_ir_text(fwd_lowered_cpu, dialect=dialect)

    bwd_cpu = ""
    if enable_backward and argnums:
        grad_fn = jax.grad(loss_fn, argnums=argnums)
        bwd_lowered_cpu = _lower_on_device(grad_fn, args, cpu_dev)
        bwd_cpu = _compiler_ir_text(bwd_lowered_cpu, dialect=dialect)

    cpu_sections = [
        f"# JAX compiler IR (backend=cpu, dialect={dialect})",
        "## Forward",
        fwd_cpu,
    ]
    if enable_backward and argnums:
        cpu_sections += ["", f"## Backward (jax.grad of scalar loss, argnums={argnums})", bwd_cpu]

    if include_llvm:
        try:
            fwd_cpu_llvm = _compiler_ir_text(fwd_lowered_cpu, dialect="llvm")
            cpu_sections += ["", "## Forward (llvm dialect)", fwd_cpu_llvm]
            if enable_backward and argnums:
                bwd_cpu_llvm = _compiler_ir_text(bwd_lowered_cpu, dialect="llvm")  # type: ignore[name-defined]
                cpu_sections += ["", "## Backward (llvm dialect)", bwd_cpu_llvm]
        except Exception:
            pass

    cpp_code = "\n".join(cpu_sections)

    # GPU IR (optional)
    cuda_code = None
    gpu_dev = _select_device("gpu")
    if gpu_dev is not None:
        try:
            fwd_lowered_gpu = _lower_on_device(fn, args, gpu_dev)
            fwd_gpu = _compiler_ir_text(fwd_lowered_gpu, dialect=dialect)

            bwd_gpu = ""
            if enable_backward and argnums:
                grad_fn = jax.grad(loss_fn, argnums=argnums)
                bwd_lowered_gpu = _lower_on_device(grad_fn, args, gpu_dev)
                bwd_gpu = _compiler_ir_text(bwd_lowered_gpu, dialect=dialect)

            gpu_sections = [
                f"# JAX compiler IR (backend=gpu, dialect={dialect})",
                "## Forward",
                fwd_gpu,
            ]
            if enable_backward and argnums:
                gpu_sections += ["", f"## Backward (jax.grad of scalar loss, argnums={argnums})", bwd_gpu]
            cuda_code = "\n".join(gpu_sections)
        except Exception:
            cuda_code = None

    return ExtractedIR(
        kernel_name=kernel_name,
        python_source=python_source,
        cpp_code=cpp_code,
        cuda_code=cuda_code,
    )


def extract_ir_pair(
    fn: Callable[..., Any],
    example_args: Sequence[Any],
    *,
    device: str = "cpu",
    dialect: str = "stablehlo",
) -> tuple[str, str]:
    """Extract a Python→IR pair suitable for training."""
    ir = extract_ir(fn, example_args, dialect=dialect)
    if device == "cuda" and ir.cuda_code:
        return (ir.python_source, ir.cuda_code)
    return (ir.python_source, ir.cpp_code)


if __name__ == "__main__":
    # Minimal smoke test
    def test_kernel(a, b):
        return a * 2.0 + b

    args = (jnp.arange(8, dtype=jnp.float32), jnp.arange(8, dtype=jnp.float32))
    ir = extract_ir(test_kernel, args, enable_backward=True, dialect="stablehlo")
    print("=== Kernel Name ===")
    print(ir.kernel_name)
    print("\n=== Python Source ===")
    print(ir.python_source)
    print("\n=== CPU IR (first 1500 chars) ===")
    print(ir.cpp_code[:1500])
    print("\n=== GPU IR available ===")
    print("Yes" if ir.cuda_code else "No")
