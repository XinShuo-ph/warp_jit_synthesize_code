"""Test IR extraction for a few JAX kernels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp

from ir_extractor import extract_ir


def add_kernel(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a + b


def dot_product(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(a * b)


def saxpy(alpha: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return alpha * x + y


def branch_kernel(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(a > 0.0, a * 2.0, a * -1.0)


def loop_kernel(a: jnp.ndarray, n: int) -> jnp.ndarray:
    # Avoid Python loops; use lax.fori_loop for JIT compatibility.
    def body(_, carry):
        return carry + a

    return jax.lax.fori_loop(0, n, body, jnp.zeros_like(a))


def run_tests():
    n = 32
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.arange(n, dtype=jnp.float32) * 2.0

    kernels = [
        ("add_kernel", add_kernel, (a, b)),
        ("dot_product", dot_product, (a, b)),
        ("saxpy", saxpy, (jnp.asarray(2.0, jnp.float32), a, b)),
        ("branch_kernel", branch_kernel, (a - 16.0,)),
        ("loop_kernel", loop_kernel, (a, 5)),
    ]

    results = []
    for name, fn, args in kernels:
        print(f"\n{'=' * 60}")
        print(f"Testing: {name}")
        print("=" * 60)

        try:
            ir = extract_ir(fn, args)
            cpu_ir = ir.cpu_code

            print(f"Python source length: {len(ir.python_source)} chars")
            print(f"CPU IR length: {len(cpu_ir)} chars")
            print(f"Has backward IR: {ir.cpu_backward_code is not None}")

            results.append((name, True, len(ir.python_source), len(cpu_ir)))
            print("\nPython source (first 300 chars):")
            print(ir.python_source[:300])
        except Exception as e:
            print(f"FAILED: {e}")
            results.append((name, False, 0, 0))

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for _, success, _, _ in results if success)
    print(f"Passed: {success_count}/{len(results)}")

    for name, success, py_len, ir_len in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}: py={py_len}, ir={ir_len}")

    return success_count == len(results)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
