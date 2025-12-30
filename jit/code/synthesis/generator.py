"""Kernel Generator - Generates varied JAX kernels programmatically."""
import random
import string
from dataclasses import dataclass
from typing import List, Optional

# JAX imports are generated in the source strings, but we don't strictly need them here 
# unless we want to validate.

@dataclass
class KernelSpec:
    """Specification for a kernel to generate."""
    name: str
    args: List[tuple]  # [(name, type_str), ...]
    body_lines: List[str]
    imports: List[str] = None


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
    """Generate a simple elementwise kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("elementwise")
    op_sym, _ = random.choice(BINARY_OPS)
    
    return f'''import jax
import jax.numpy as jnp

@jax.jit
def {name}(a, b):
    # Elementwise {op_sym}
    return a {op_sym} b
'''


def generate_scalar_array_op(seed: int = None) -> str:
    """Generate a kernel with scalar and array operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("scalar_arr")
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    
    return f'''import jax
import jax.numpy as jnp

@jax.jit
def {name}(alpha, x, y):
    # Scalar-array operations
    return alpha {op1_sym} x {op2_sym} y
'''


def generate_unary_kernel(seed: int = None) -> str:
    """Generate a kernel with unary operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("unary")
    op_func, _ = random.choice(UNARY_OPS)
    
    return f'''import jax
import jax.numpy as jnp

@jax.jit
def {name}(a):
    # Unary operation {op_func}
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
    
    return f'''import jax
import jax.numpy as jnp

@jax.jit
def {name}(a):
    # Branching with where
    # if a > {threshold}: a {op1_sym} {const1} else: a {op2_sym} {const2}
    return jnp.where(a > {threshold}, a {op1_sym} {const1}, a {op2_sym} {const2})
'''


def generate_loop_kernel(seed: int = None) -> str:
    """Generate a kernel with a loop using lax.scan or fori_loop."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("loop")
    op_sym, _ = random.choice(BINARY_OPS)
    
    # Simulating:
    # total = 0
    # for i in range(n):
    #    total = total {op} a
    # This implies repeated application.
    # To keep it consistent with the Warp kernel which did `total = total op a[tid]` n times:
    # Warp: `for i in range(n): total = total {op_sym} a[tid]`
    # This means adding `a[tid]` n times to total.
    # In JAX: `val = jax.lax.fori_loop(0, n, lambda i, v: v {op_sym} a, 0.0)`
    
    return f'''import jax
import jax.numpy as jnp

@jax.jit
def {name}(a, n):
    # Loop operation
    def body_fun(i, val):
        return val {op_sym} a
    
    # We apply the body_fun n times starting from 0.0
    # Since 'a' is an array, this operates elementwise across 'a'
    init_val = jnp.zeros_like(a)
    return jax.lax.fori_loop(0, n, body_fun, init_val)
'''


def generate_reduction_kernel(seed: int = None) -> str:
    """Generate a reduction kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("reduce")
    
    return f'''import jax
import jax.numpy as jnp

@jax.jit
def {name}(a):
    # Reduction (sum)
    return jnp.sum(a)
'''


def generate_vector_kernel(seed: int = None) -> str:
    """Generate a kernel with vector operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("vec")
    ops = ["jnp.dot", "jnp.linalg.norm"] # replacing length with norm
    op = random.choice(ops)
    
    # Warp vec3 kernel operated on arrays of vec3.
    # JAX: Inputs should be (N, 3).
    # Operations should be mapped or vectorized along axis 1.
    
    if op == "jnp.dot":
        # dot product of two (N, 3) arrays row-wise -> (N,)
        return f'''import jax
import jax.numpy as jnp

@jax.jit
def {name}(a, b):
    # Vector dot product (row-wise for N x 3 arrays)
    # a and b are expected to be shape (N, 3)
    # contract over last axis
    return jnp.sum(a * b, axis=-1)
'''
    else:
        # length/norm
        return f'''import jax
import jax.numpy as jnp

@jax.jit
def {name}(a):
    # Vector length (row-wise for N x 3 arrays)
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
    
    return f'''import jax
import jax.numpy as jnp

@jax.jit
def {name}(a, b):
    # Multi-statement
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
    
    return f'''import jax
import jax.numpy as jnp

@jax.jit
def {name}(a):
    # Nested branching
    # if a > {t1}:
    #   if a > {t2}: a * 3.0
    #   else: a * 2.0
    # else: a * 0.5
    
    inner_branch = jnp.where(a > {t2}, a * 3.0, a * 2.0)
    return jnp.where(a > {t1}, inner_branch, a * 0.5)
'''


def generate_compound_kernel(seed: int = None) -> str:
    """Generate a kernel with compound operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("compound")
    
    return f'''import jax
import jax.numpy as jnp

@jax.jit
def {name}(a, b, scale):
    # Compound operations
    x = a
    y = b
    result = (x + y) * scale
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
