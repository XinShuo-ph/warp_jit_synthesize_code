"""Kernel Generator - Generates varied JAX kernels programmatically.

Each generator returns a Python source string that defines:
- a pure function kernel: `def <name>(...): ...`
- `EXAMPLE_ARGS`: a tuple of example inputs used for lowering/compilation

The downstream pipeline loads this module and extracts JAX compiler IR for:
- forward computation
- backward computation (via `jax.grad`)
"""

from __future__ import annotations

import random
import string
from typing import List


# Binary operations
BINARY_OPS = [
    "+",
    "-",
    "*",
]

# Unary operations
UNARY_OPS = [
    "jnp.sqrt",
    "jnp.abs",
    "jnp.sin",
    "jnp.cos",
]


def random_name(prefix: str = "kernel") -> str:
    """Generate a random kernel name."""
    suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f"{prefix}_{suffix}"


def generate_simple_elementwise(seed: int = None) -> str:
    """Generate a simple elementwise array kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("elementwise")
    op_sym = random.choice(BINARY_OPS)

    return f'''import jax
import jax.numpy as jnp

def {name}(a, b):
    return a {op_sym} b

EXAMPLE_ARGS = (
    jnp.arange(8, dtype=jnp.float32),
    jnp.arange(8, dtype=jnp.float32),
)
'''


def generate_scalar_array_op(seed: int = None) -> str:
    """Generate a kernel with scalar and array operations (SAXPY-like)."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("scalar_arr")
    op1_sym = random.choice(BINARY_OPS)
    op2_sym = random.choice(BINARY_OPS)

    return f'''import jax
import jax.numpy as jnp

def {name}(alpha, x, y):
    return alpha {op1_sym} x {op2_sym} y

EXAMPLE_ARGS = (
    jnp.array(2.0, dtype=jnp.float32),
    jnp.arange(8, dtype=jnp.float32),
    (jnp.arange(8, dtype=jnp.float32) * 10.0),
)
'''


def generate_unary_kernel(seed: int = None) -> str:
    """Generate a kernel with unary operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("unary")
    op_func = random.choice(UNARY_OPS)

    return f'''import jax
import jax.numpy as jnp

def {name}(a):
    return {op_func}(a)

EXAMPLE_ARGS = (
    jnp.linspace(0.1, 1.0, 8, dtype=jnp.float32),
)
'''


def generate_branch_kernel(seed: int = None) -> str:
    """Generate a kernel with branching."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("branch")
    threshold = round(random.uniform(-1.0, 1.0), 2)
    op1_sym = random.choice(BINARY_OPS)
    op2_sym = random.choice(BINARY_OPS)
    const1 = round(random.uniform(0.5, 3.0), 1)
    const2 = round(random.uniform(0.5, 3.0), 1)
    
    return f'''import jax
import jax.numpy as jnp

def {name}(a):
    val = a
    return jnp.where(val > {threshold}, val {op1_sym} {const1}, val {op2_sym} {const2})

EXAMPLE_ARGS = (
    jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float32),
)
'''


def generate_loop_kernel(seed: int = None) -> str:
    """Generate a kernel with a loop."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("loop")
    op_sym = random.choice(BINARY_OPS)

    return f'''import jax
import jax.numpy as jnp
from jax import lax

def {name}(a, n):
    def body(i, total):
        return total {op_sym} a
    total0 = jnp.zeros_like(a)
    return lax.fori_loop(0, n, body, total0)

EXAMPLE_ARGS = (
    jnp.arange(8, dtype=jnp.float32),
    5,
)
'''


def generate_reduction_kernel(seed: int = None) -> str:
    """Generate a reduction kernel (sum)."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("reduce")
    
    return f'''import jax
import jax.numpy as jnp

def {name}(a):
    return jnp.sum(a)

EXAMPLE_ARGS = (
    jnp.arange(8, dtype=jnp.float32),
)
'''


def generate_vector_kernel(seed: int = None) -> str:
    """Generate a kernel with vector operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("vec")
    ops = ["wp.dot", "wp.length", "wp.cross"]
    op = random.choice(ops[:2])  # dot or length
    
    if op == "wp.dot":
        return f'''import jax
import jax.numpy as jnp

def {name}(a, b):
    # a, b: (N, 3)
    return jnp.sum(a * b, axis=-1)

EXAMPLE_ARGS = (
    jnp.arange(24, dtype=jnp.float32).reshape(8, 3),
    (jnp.arange(24, dtype=jnp.float32).reshape(8, 3) * 0.5),
)
'''

    return f'''import jax
import jax.numpy as jnp

def {name}(a):
    # a: (N, 3)
    return jnp.sqrt(jnp.sum(a * a, axis=-1))

EXAMPLE_ARGS = (
    jnp.arange(24, dtype=jnp.float32).reshape(8, 3),
)
'''


def generate_multi_statement_kernel(seed: int = None) -> str:
    """Generate a kernel with multiple statements."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("multi")
    op1_sym = random.choice(BINARY_OPS)
    op2_sym = random.choice(BINARY_OPS)
    unary_op = random.choice(UNARY_OPS)
    
    return f'''import jax
import jax.numpy as jnp

def {name}(a, b):
    temp1 = a {op1_sym} b
    temp2 = {unary_op}(temp1)
    return temp2 {op2_sym} a

EXAMPLE_ARGS = (
    jnp.linspace(0.1, 1.0, 8, dtype=jnp.float32),
    jnp.linspace(1.0, 2.0, 8, dtype=jnp.float32),
)
'''


def generate_nested_branch_kernel(seed: int = None) -> str:
    """Generate a kernel with nested branches."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("nested")
    t1 = round(random.uniform(-0.5, 0.5), 2)
    t2 = round(random.uniform(0.5, 1.5), 2)
    
    return f'''import jax
import jax.numpy as jnp

def {name}(a):
    val = a
    inner = jnp.where(val > {t2}, val * 3.0, val * 2.0)
    return jnp.where(val > {t1}, inner, val * 0.5)

EXAMPLE_ARGS = (
    jnp.linspace(-1.0, 2.0, 8, dtype=jnp.float32),
)
'''


def generate_compound_kernel(seed: int = None) -> str:
    """Generate a kernel with compound operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("compound")
    
    return f'''import jax
import jax.numpy as jnp

def {name}(a, b, scale):
    x = a
    y = b
    result = (x + y) * scale
    result = result - jnp.floor(result)
    return jnp.abs(result)

EXAMPLE_ARGS = (
    jnp.linspace(0.0, 1.0, 8, dtype=jnp.float32),
    jnp.linspace(1.0, 2.0, 8, dtype=jnp.float32),
    jnp.array(1.3, dtype=jnp.float32),
)
'''


# All generator functions
GENERATORS = [
    generate_simple_elementwise,
    generate_scalar_array_op,
    generate_unary_kernel,
    generate_branch_kernel,
    generate_loop_kernel,
    generate_reduction_kernel,
    generate_vector_kernel,
    generate_multi_statement_kernel,
    generate_nested_branch_kernel,
    generate_compound_kernel,
]


def generate_random_kernel(seed: int = None) -> str:
    """Generate a random kernel from available templates."""
    if seed is not None:
        random.seed(seed)
    generator = random.choice(GENERATORS)
    return generator(seed)


def generate_kernel_batch(count: int, base_seed: int = 42) -> List[str]:
    """Generate a batch of unique kernels."""
    kernels = []
    for i in range(count):
        kernel = generate_random_kernel(base_seed + i)
        kernels.append(kernel)
    return kernels


if __name__ == "__main__":
    # Test kernel generation
    print("=== Testing Kernel Generator ===\n")
    
    for i, gen_func in enumerate(GENERATORS):
        print(f"--- Generator: {gen_func.__name__} ---")
        kernel_src = gen_func(seed=42 + i)
        print(kernel_src)
    
    print("\n=== Batch Generation Test ===")
    batch = generate_kernel_batch(5, base_seed=100)
    for i, k in enumerate(batch):
        print(f"Kernel {i+1}:")
        print(k)
