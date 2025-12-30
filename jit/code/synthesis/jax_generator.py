"""JAX Kernel Generator - Generates varied JAX functions programmatically.

This module generates JAX-based functions equivalent to the Warp kernel types.
JAX uses vectorized operations and JIT compilation via XLA.
"""
import random
import string
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KernelSpec:
    """Specification for a JAX function to generate."""
    name: str
    args: List[tuple]  # [(name, type_str), ...]
    body_lines: List[str]
    imports: List[str] = None


# Type definitions for function arguments
ARRAY_TYPES = [
    "jnp.ndarray",  # float32 array
    "jnp.ndarray",  # int32 array
]

SCALAR_TYPES = ["float", "int"]

# Binary operations
BINARY_OPS = [
    ("+", "jnp.add"),
    ("-", "jnp.subtract"),
    ("*", "jnp.multiply"),
]

# Unary operations
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
    """Generate a simple elementwise JAX function."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("elementwise")
    op_sym, _ = random.choice(BINARY_OPS)
    
    return f'''@jax.jit
def {name}(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Elementwise operation on two arrays."""
    return a {op_sym} b
'''


def generate_scalar_array_op(seed: int = None) -> str:
    """Generate a function with scalar and array operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("scalar_arr")
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    
    return f'''@jax.jit
def {name}(alpha: float, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Scalar-array combined operation."""
    return alpha {op1_sym} x {op2_sym} y
'''


def generate_unary_kernel(seed: int = None) -> str:
    """Generate a function with unary operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("unary")
    op_func, _ = random.choice(UNARY_OPS)
    
    return f'''@jax.jit
def {name}(a: jnp.ndarray) -> jnp.ndarray:
    """Unary math operation."""
    return {op_func}(a)
'''


def generate_branch_kernel(seed: int = None) -> str:
    """Generate a function with branching using jnp.where."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("branch")
    threshold = round(random.uniform(-1.0, 1.0), 2)
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    const1 = round(random.uniform(0.5, 3.0), 1)
    const2 = round(random.uniform(0.5, 3.0), 1)
    
    return f'''@jax.jit
def {name}(a: jnp.ndarray) -> jnp.ndarray:
    """Conditional operation using jnp.where."""
    return jnp.where(
        a > {threshold},
        a {op1_sym} {const1},
        a {op2_sym} {const2}
    )
'''


def generate_loop_kernel(seed: int = None) -> str:
    """Generate a function with a loop using jax.lax.fori_loop."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("loop")
    op_sym, _ = random.choice(BINARY_OPS)
    
    return f'''@jax.jit
def {name}(a: jnp.ndarray, n: int) -> jnp.ndarray:
    """Loop-based accumulation using jax.lax.fori_loop."""
    def body_fn(i, total):
        return total {op_sym} a
    return jax.lax.fori_loop(0, n, body_fn, jnp.zeros_like(a))
'''


def generate_reduction_kernel(seed: int = None) -> str:
    """Generate a reduction function."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("reduce")
    
    return f'''@jax.jit
def {name}(a: jnp.ndarray) -> jnp.ndarray:
    """Reduction (sum) over array."""
    return jnp.sum(a)
'''


def generate_vector_kernel(seed: int = None) -> str:
    """Generate a function with vector operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("vec")
    ops = ["dot", "norm"]
    op = random.choice(ops)
    
    if op == "dot":
        return f'''@jax.jit
def {name}(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Batch dot product of 3D vectors."""
    return jnp.sum(a * b, axis=-1)
'''
    else:
        return f'''@jax.jit
def {name}(a: jnp.ndarray) -> jnp.ndarray:
    """Compute vector norms."""
    return jnp.linalg.norm(a, axis=-1)
'''


def generate_multi_statement_kernel(seed: int = None) -> str:
    """Generate a function with multiple statements."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("multi")
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    unary_op, _ = random.choice(UNARY_OPS)
    
    return f'''@jax.jit
def {name}(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Multi-step computation."""
    temp1 = a {op1_sym} b
    temp2 = {unary_op}(temp1)
    return temp2 {op2_sym} a
'''


def generate_nested_branch_kernel(seed: int = None) -> str:
    """Generate a function with nested branches."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("nested")
    t1 = round(random.uniform(-0.5, 0.5), 2)
    t2 = round(random.uniform(0.5, 1.5), 2)
    
    return f'''@jax.jit
def {name}(a: jnp.ndarray) -> jnp.ndarray:
    """Nested conditional using jnp.where."""
    return jnp.where(
        a > {t1},
        jnp.where(a > {t2}, a * 3.0, a * 2.0),
        a * 0.5
    )
'''


def generate_compound_kernel(seed: int = None) -> str:
    """Generate a function with compound operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("compound")
    
    return f'''@jax.jit
def {name}(a: jnp.ndarray, b: jnp.ndarray, scale: float) -> jnp.ndarray:
    """Compound operation with multiple steps."""
    result = (a + b) * scale
    result = result - jnp.floor(result)
    return jnp.abs(result)
'''


def generate_matmul_kernel(seed: int = None) -> str:
    """Generate a matrix multiplication function."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("matmul")
    
    return f'''@jax.jit
def {name}(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Matrix multiplication."""
    return jnp.matmul(a, b)
'''


def generate_softmax_kernel(seed: int = None) -> str:
    """Generate a softmax function."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("softmax")
    
    return f'''@jax.jit
def {name}(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable softmax."""
    x_max = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)
'''


def generate_scan_kernel(seed: int = None) -> str:
    """Generate a function using jax.lax.scan for sequential operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("scan")
    
    return f'''@jax.jit
def {name}(x: jnp.ndarray) -> jnp.ndarray:
    """Cumulative sum using jax.lax.scan."""
    def scan_fn(carry, elem):
        new_carry = carry + elem
        return new_carry, new_carry
    _, result = jax.lax.scan(scan_fn, 0.0, x)
    return result
'''


def generate_vmap_kernel(seed: int = None) -> str:
    """Generate a function demonstrating vmap for batching."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("vmap")
    
    return f'''def _{name}_single(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Single instance computation."""
    return jnp.dot(a, b)

@jax.jit
def {name}(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Batched dot product using vmap."""
    return jax.vmap(_{name}_single)(a, b)
'''


def generate_grad_kernel(seed: int = None) -> str:
    """Generate a function with gradient computation."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("grad")
    
    return f'''def _{name}_loss(x: jnp.ndarray) -> float:
    """Loss function: sum of squared values."""
    return jnp.sum(x ** 2)

@jax.jit
def {name}(x: jnp.ndarray) -> jnp.ndarray:
    """Compute gradient of sum of squares."""
    return jax.grad(_{name}_loss)(x)
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
    generate_matmul_kernel,
    generate_softmax_kernel,
    generate_scan_kernel,
    generate_vmap_kernel,
    generate_grad_kernel,
]


def generate_random_kernel(seed: int = None) -> str:
    """Generate a random JAX function from available templates."""
    if seed is not None:
        random.seed(seed)
    generator = random.choice(GENERATORS)
    return generator(seed)


def generate_kernel_batch(count: int, base_seed: int = 42) -> List[str]:
    """Generate a batch of unique JAX functions."""
    kernels = []
    for i in range(count):
        kernel = generate_random_kernel(base_seed + i)
        kernels.append(kernel)
    return kernels


if __name__ == "__main__":
    # Test kernel generation
    print("=== Testing JAX Kernel Generator ===\n")
    
    for i, gen_func in enumerate(GENERATORS):
        print(f"--- Generator: {gen_func.__name__} ---")
        kernel_src = gen_func(seed=42 + i)
        print(kernel_src)
    
    print("\n=== Batch Generation Test ===")
    batch = generate_kernel_batch(5, base_seed=100)
    for i, k in enumerate(batch):
        print(f"Kernel {i+1}:")
        print(k)
