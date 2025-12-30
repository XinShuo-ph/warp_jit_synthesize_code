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


# Binary operations (symbol only; we generate expression strings)
BINARY_OPS = ["+", "-", "*"]

# Unary operations (JAX-friendly)
UNARY_OPS = ["jnp.sqrt", "jnp.abs", "jnp.sin", "jnp.cos"]


def random_name(prefix: str = "kernel") -> str:
    """Generate a random kernel name."""
    suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f"{prefix}_{suffix}"


def generate_simple_elementwise(seed: int = None) -> str:
    """Generate a simple elementwise kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("elementwise")
    op_sym = random.choice(BINARY_OPS)
    
    return f'''def {name}(a, b):
    return a {op_sym} b
'''


def generate_scalar_array_op(seed: int = None) -> str:
    """Generate a kernel with scalar and array operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("scalar_arr")
    op1_sym = random.choice(BINARY_OPS)
    op2_sym = random.choice(BINARY_OPS)
    
    return f'''def {name}(alpha, x, y):
    return (alpha {op1_sym} x) {op2_sym} y
'''


def generate_unary_kernel(seed: int = None) -> str:
    """Generate a kernel with unary operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("unary")
    op_func = random.choice(UNARY_OPS)
    
    return f'''def {name}(a):
    return {op_func}(a)
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
    
    return f'''def {name}(a):
    return jnp.where(a > {threshold}, a {op1_sym} {const1}, a {op2_sym} {const2})
'''


def generate_loop_kernel(seed: int = None) -> str:
    """Generate a kernel with a loop (uses lax.fori_loop)."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("loop")
    op_sym = random.choice(BINARY_OPS)
    
    # Loop accumulates an array accumulator (same shape as a)
    return f'''def {name}(a, n):
    def body(i, acc):
        return acc {op_sym} a
    init = jnp.zeros_like(a)
    return lax.fori_loop(0, n, body, init)
'''


def generate_reduction_kernel(seed: int = None) -> str:
    """Generate a reduction kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("reduce")
    
    return f'''def {name}(a):
    return jnp.sum(a)
'''


def generate_vector_kernel(seed: int = None) -> str:
    """Generate a kernel with vector operations on (N, 3) arrays."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("vec")
    ops = ["dot", "length"]
    op = random.choice(ops)
    
    if op == "dot":
        return f'''def {name}(a, b):
    return jnp.sum(a * b, axis=-1)
'''
    else:
        return f'''def {name}(a):
    return jnp.sqrt(jnp.sum(a * a, axis=-1))
'''


def generate_multi_statement_kernel(seed: int = None) -> str:
    """Generate a kernel with multiple statements."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("multi")
    op1_sym = random.choice(BINARY_OPS)
    op2_sym = random.choice(BINARY_OPS)
    unary_op = random.choice(UNARY_OPS)
    
    return f'''def {name}(a, b):
    temp1 = a {op1_sym} b
    temp2 = {unary_op}(temp1)
    return temp2 {op2_sym} a
'''


def generate_nested_branch_kernel(seed: int = None) -> str:
    """Generate a kernel with nested branches."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("nested")
    t1 = round(random.uniform(-0.5, 0.5), 2)
    t2 = round(random.uniform(0.5, 1.5), 2)
    
    return f'''def {name}(a):
    inner = jnp.where(a > {t2}, a * 3.0, a * 2.0)
    return jnp.where(a > {t1}, inner, a * 0.5)
'''


def generate_compound_kernel(seed: int = None) -> str:
    """Generate a kernel with compound operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("compound")
    
    return f'''def {name}(a, b, scale):
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
