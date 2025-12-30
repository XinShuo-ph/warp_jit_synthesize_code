"""Test IR extraction with various JAX kernel types."""

import sys

sys.path.insert(0, "/workspace/jit/code/extraction")

import jax
import jax.numpy as jnp
from jax import lax

from ir_extractor import extract_ir


def add_kernel(a, b):
    return a + b


def dot_product(a, b):
    return jnp.sum(a * b)


def saxpy(alpha, x, y):
    return alpha * x + y


def branch_kernel(a):
    return jnp.where(a > 0.0, a * 2.0, a * -1.0)


def loop_kernel(a, n):
    def body(i, total):
        return total + a

    return lax.fori_loop(0, n, body, jnp.zeros_like(a))


def vec_kernel(a, b):
    # a, b: (N, 3) → (N,)
    return jnp.sum(a * b, axis=-1)


def run_tests():
    kernels = [
        ("add_kernel", add_kernel, (jnp.arange(8, dtype=jnp.float32), jnp.arange(8, dtype=jnp.float32))),
        ("dot_product", dot_product, (jnp.arange(8, dtype=jnp.float32), jnp.arange(8, dtype=jnp.float32))),
        ("saxpy", saxpy, (jnp.array(2.0, dtype=jnp.float32), jnp.arange(8, dtype=jnp.float32), jnp.arange(8, dtype=jnp.float32))),
        ("branch_kernel", branch_kernel, (jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float32),)),
        ("loop_kernel", loop_kernel, (jnp.arange(8, dtype=jnp.float32), 5)),
        ("vec_kernel", vec_kernel, (jnp.arange(24, dtype=jnp.float32).reshape(8, 3), jnp.arange(24, dtype=jnp.float32).reshape(8, 3))),
    ]

    results = []
    for name, fn, example_args in kernels:
        print(f"\n{'=' * 60}")
        print(f"Testing: {name}")
        print("=" * 60)

        try:
            ir = extract_ir(fn, example_args, kernel_name=name, python_source=f"# {name} defined in test file")
            cpu_ir = ir.cpp_code

            print(f"CPU IR length: {len(cpu_ir)} chars")
            print(f"Has forward section: {'## Forward' in cpu_ir}")
            print(f"Has backward section: {'## Backward' in cpu_ir}")
            print(f"Has stablehlo/hlö ops: {('stablehlo.' in cpu_ir) or ('hlo.' in cpu_ir)}")

            results.append((name, True, len(cpu_ir)))

            print("\nCPU IR (first 300 chars):")
            print(cpu_ir[:300])

        except Exception as e:
            print(f"FAILED: {e}")
            results.append((name, False, 0))

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for _, success, _ in results if success)
    print(f"Passed: {success_count}/{len(results)}")

    for name, success, ir_len in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}: ir={ir_len}")

    return success_count == len(results)


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
