"""Kernel Generator - Generates varied JAX kernels programmatically."""
import random
import string
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KernelSpec:
    """Specification for a kernel to generate."""
    name: str
    args: List[tuple]  # [(name, type_str), ...]
    body_lines: List[str]
    imports: List[str] = None


# Type definitions for kernel arguments - JAX uses numpy-like arrays
ARRAY_TYPES = [
    "jnp.ndarray",  # float32 by default
    "jnp.ndarray",  # int32
    "jnp.ndarray",  # vectors (3D)
    "jnp.ndarray",  # vectors (2D)
]

SCALAR_TYPES = ["float", "int"]

# Binary operations - JAX uses numpy operations
BINARY_OPS = [
    ("+", "add"),
    ("-", "subtract"),
    ("*", "multiply"),
]

# Unary operations - JAX math functions
UNARY_OPS = [
    ("jnp.sqrt", "sqrt"),
    ("jnp.abs", "abs"),
    ("jnp.sin", "sin"),
    ("jnp.cos", "cos"),
]


def random_name(prefix: str = "kernel") -> str:
    """Generate a random kernel name."""
    suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f"{prefix}_{suffix}"


def generate_simple_elementwise(seed: int = None) -> str:
    """Generate a simple elementwise kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("elementwise")
    op_sym, _ = random.choice(BINARY_OPS)
    
    return f'''def {name}(a, b):
    """Elementwise operation on two arrays."""
    return a {op_sym} b
'''


def generate_scalar_array_op(seed: int = None) -> str:
    """Generate a kernel with scalar and array operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("scalar_arr")
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    
    return f'''def {name}(alpha, x, y):
    """Scalar and array operations."""
    return alpha {op1_sym} x {op2_sym} y
'''


def generate_unary_kernel(seed: int = None) -> str:
    """Generate a kernel with unary operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("unary")
    op_func, _ = random.choice(UNARY_OPS)
    
    return f'''def {name}(a):
    """Unary operation on array."""
    return {op_func}(a)
'''


def generate_branch_kernel(seed: int = None) -> str:
    """Generate a kernel with branching using jnp.where."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("branch")
    threshold = round(random.uniform(-1.0, 1.0), 2)
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    const1 = round(random.uniform(0.5, 3.0), 1)
    const2 = round(random.uniform(0.5, 3.0), 1)
    
    return f'''def {name}(a):
    """Conditional operation using jnp.where."""
    return jnp.where(a > {threshold}, a {op1_sym} {const1}, a {op2_sym} {const2})
'''


def generate_loop_kernel(seed: int = None) -> str:
    """Generate a kernel with a reduction (replaces explicit loop)."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("loop")
    op_sym, _ = random.choice(BINARY_OPS)
    
    # JAX doesn't use explicit loops - we use reductions
    # Use a fixed iteration count to avoid concretization issues
    n_iters = random.choice([3, 5, 10])
    
    return f'''def {name}(a):
    """Reduction operation simulating loop accumulation."""
    # Accumulate n={n_iters} times
    result = a
    for _ in range({n_iters}):
        result = result {op_sym} a
    return result
'''


def generate_reduction_kernel(seed: int = None) -> str:
    """Generate a reduction kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("reduce")
    
    return f'''def {name}(a):
    """Sum reduction over array."""
    return jnp.sum(a)
'''


def generate_vector_kernel(seed: int = None) -> str:
    """Generate a kernel with vector operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("vec")
    ops = ["dot", "norm"]
    op = random.choice(ops)
    
    if op == "dot":
        return f'''def {name}(a, b):
    """Dot product along last axis."""
    # Assuming last dimension is vector dimension
    return jnp.sum(a * b, axis=-1)
'''
    else:
        return f'''def {name}(a):
    """Vector norm computation."""
    return jnp.linalg.norm(a, axis=-1)
'''


def generate_multi_statement_kernel(seed: int = None) -> str:
    """Generate a kernel with multiple statements."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("multi")
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    unary_op, _ = random.choice(UNARY_OPS)
    
    return f'''def {name}(a, b):
    """Multi-statement computation."""
    temp1 = a {op1_sym} b
    temp2 = {unary_op}(temp1)
    result = temp2 {op2_sym} a
    return result
'''


def generate_nested_branch_kernel(seed: int = None) -> str:
    """Generate a kernel with nested branches."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("nested")
    t1 = round(random.uniform(-0.5, 0.5), 2)
    t2 = round(random.uniform(0.5, 1.5), 2)
    
    return f'''def {name}(a):
    """Nested conditional using jnp.where."""
    return jnp.where(a > {t1}, 
                     jnp.where(a > {t2}, a * 3.0, a * 2.0),
                     a * 0.5)
'''


def generate_compound_kernel(seed: int = None) -> str:
    """Generate a kernel with compound operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("compound")
    
    return f'''def {name}(a, b, scale):
    """Compound operations with multiple steps."""
    result = (a + b) * scale
    result = result - jnp.floor(result)
    return jnp.abs(result)
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
