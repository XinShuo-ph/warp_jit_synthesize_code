"""Kernel Generator - Generates varied JAX kernels programmatically.

Each generator returns a Python source string that defines:
- a function `def <kernel_name>(...): ...`
- an `EXAMPLE_ARGS` tuple with deterministic inputs for compilation
- optionally a `GRAD_ARGNUMS` tuple for which args to differentiate

The synthesis pipeline loads the module and lowers the function via JAX.
"""

import random
import string
from typing import List


def random_name(prefix: str = "kernel") -> str:
    """Generate a random kernel name."""
    suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f"{prefix}_{suffix}"


def generate_simple_elementwise(seed: int = None) -> str:
    """Generate a simple elementwise kernel (array op)."""
    if seed is not None:
        random.seed(seed)

    name = random_name("elementwise")

    op_sym = random.choice(["+", "-", "*"])

    return f'''def {name}(a, b):
    return a {op_sym} b


EXAMPLE_ARGS = (
    jnp.arange(128, dtype=jnp.float32),
    jnp.arange(128, dtype=jnp.float32) * 2.0,
)

GRAD_ARGNUMS = (0, 1)
'''


def generate_scalar_array_op(seed: int = None) -> str:
    """Generate a kernel with scalar and array operations (SAXPY-like)."""
    if seed is not None:
        random.seed(seed)

    name = random_name("scalar_arr")
    op1_sym = random.choice(["+", "-", "*"])
    op2_sym = random.choice(["+", "-", "*"])

    return f'''def {name}(alpha, x, y):
    return (alpha {op1_sym} x) {op2_sym} y


EXAMPLE_ARGS = (
    jnp.asarray(2.0, dtype=jnp.float32),
    jnp.arange(128, dtype=jnp.float32),
    jnp.arange(128, dtype=jnp.float32) * 10.0,
)

GRAD_ARGNUMS = (0, 1, 2)
'''


def generate_unary_kernel(seed: int = None) -> str:
    """Generate a kernel with unary operations."""
    if seed is not None:
        random.seed(seed)

    name = random_name("unary")
    op_func = random.choice(["jnp.sqrt", "jnp.abs", "jnp.sin", "jnp.cos"])

    return f'''def {name}(a):
    return {op_func}(a)


EXAMPLE_ARGS = (jnp.linspace(-1.0, 1.0, 128, dtype=jnp.float32),)
GRAD_ARGNUMS = (0,)
'''


def generate_branch_kernel(seed: int = None) -> str:
    """Generate a kernel with branching."""
    if seed is not None:
        random.seed(seed)

    name = random_name("branch")
    threshold = round(random.uniform(-1.0, 1.0), 2)
    op1_sym = random.choice(["+", "-", "*"])
    op2_sym = random.choice(["+", "-", "*"])
    const1 = round(random.uniform(0.5, 3.0), 1)
    const2 = round(random.uniform(0.5, 3.0), 1)

    return f'''def {name}(a):
    return jnp.where(a > {threshold}, a {op1_sym} {const1}, a {op2_sym} {const2})


EXAMPLE_ARGS = (jnp.linspace(-2.0, 2.0, 128, dtype=jnp.float32),)
GRAD_ARGNUMS = (0,)
'''


def generate_loop_kernel(seed: int = None) -> str:
    """Generate a kernel with a loop."""
    if seed is not None:
        random.seed(seed)

    name = random_name("loop")
    op_sym = random.choice(["+", "*"])
    n = random.randint(2, 8)

    # Use lax.fori_loop for JIT.
    return f'''def {name}(a):
    def body(_, carry):
        return carry {op_sym} a
    return lax.fori_loop(0, {n}, body, jnp.zeros_like(a) + (1.0 if "{op_sym}" == "*" else 0.0))


EXAMPLE_ARGS = (jnp.arange(128, dtype=jnp.float32),)
GRAD_ARGNUMS = (0,)
'''


def generate_reduction_kernel(seed: int = None) -> str:
    """Generate an atomic reduction kernel."""
    if seed is not None:
        random.seed(seed)

    name = random_name("reduce")

    return f'''def {name}(a):
    return jnp.sum(a)


EXAMPLE_ARGS = (jnp.arange(128, dtype=jnp.float32),)
GRAD_ARGNUMS = (0,)
'''


def generate_vector_kernel(seed: int = None) -> str:
    """Generate a kernel with vector operations."""
    if seed is not None:
        random.seed(seed)

    name = random_name("vec")

    op = random.choice(["dot", "length"])
    if op == "dot":
        return f'''def {name}(a, b):
    return jnp.sum(a * b, axis=-1)


EXAMPLE_ARGS = (
    jnp.arange(128 * 3, dtype=jnp.float32).reshape(128, 3),
    (jnp.arange(128 * 3, dtype=jnp.float32).reshape(128, 3) * 0.5),
)

GRAD_ARGNUMS = (0, 1)
'''

    return f'''def {name}(a):
    return jnp.linalg.norm(a, axis=-1)


EXAMPLE_ARGS = (jnp.arange(128 * 3, dtype=jnp.float32).reshape(128, 3),)
GRAD_ARGNUMS = (0,)
'''


def generate_multi_statement_kernel(seed: int = None) -> str:
    """Generate a kernel with multiple statements."""
    if seed is not None:
        random.seed(seed)

    name = random_name("multi")
    op1_sym = random.choice(["+", "-", "*"])
    op2_sym = random.choice(["+", "-", "*"])
    unary_op = random.choice(["jnp.sqrt", "jnp.abs", "jnp.sin", "jnp.cos"])

    return f'''def {name}(a, b):
    temp1 = a {op1_sym} b
    temp2 = {unary_op}(temp1)
    return temp2 {op2_sym} a


EXAMPLE_ARGS = (
    jnp.linspace(0.1, 2.0, 128, dtype=jnp.float32),
    jnp.linspace(-1.0, 1.0, 128, dtype=jnp.float32),
)

GRAD_ARGNUMS = (0, 1)
'''


def generate_nested_branch_kernel(seed: int = None) -> str:
    """Generate a kernel with nested branches."""
    if seed is not None:
        random.seed(seed)

    name = random_name("nested")
    t1 = round(random.uniform(-0.5, 0.5), 2)
    t2 = round(random.uniform(0.5, 1.5), 2)

    return f'''def {name}(a):
    return jnp.where(
        a > {t1},
        jnp.where(a > {t2}, a * 3.0, a * 2.0),
        a * 0.5,
    )


EXAMPLE_ARGS = (jnp.linspace(-2.0, 2.0, 128, dtype=jnp.float32),)
GRAD_ARGNUMS = (0,)
'''


def generate_compound_kernel(seed: int = None) -> str:
    """Generate a kernel with compound operations."""
    if seed is not None:
        random.seed(seed)

    name = random_name("compound")

    scale = round(random.uniform(0.2, 3.0), 2)

    return f'''def {name}(a, b):
    result = (a + b) * {scale}
    result = result - jnp.floor(result)
    return jnp.abs(result)


EXAMPLE_ARGS = (
    jnp.linspace(-3.0, 3.0, 128, dtype=jnp.float32),
    jnp.linspace(0.5, 1.5, 128, dtype=jnp.float32),
)

GRAD_ARGNUMS = (0, 1)
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
