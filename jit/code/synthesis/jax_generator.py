"""JAX Kernel Generator - Generates varied JAX functions programmatically."""
import random
import string
from dataclasses import dataclass
from typing import List

@dataclass
class KernelSpec:
    """Specification for a kernel to generate."""
    name: str
    args: List[tuple]
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
    
    return f'''@jax.jit
def {name}(a, b):
    return a {op_sym} b
'''

def generate_scalar_array_op(seed: int = None) -> str:
    """Generate a kernel with scalar and array operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("scalar_arr")
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    
    return f'''@jax.jit
def {name}(alpha, x, y):
    return alpha {op1_sym} x {op2_sym} y
'''

def generate_unary_kernel(seed: int = None) -> str:
    """Generate a kernel with unary operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("unary")
    op_func, _ = random.choice(UNARY_OPS)
    
    return f'''@jax.jit
def {name}(a):
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
    
    return f'''@jax.jit
def {name}(a):
    return jnp.where(a > {threshold}, a {op1_sym} {const1}, a {op2_sym} {const2})
'''

def generate_loop_kernel(seed: int = None) -> str:
    """Generate a kernel with a loop using jax.lax.fori_loop."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("loop")
    op_sym, _ = random.choice(BINARY_OPS)
    
    return f'''@jax.jit
def {name}(a, n):
    def body_fun(i, val):
        return val {op_sym} a
    # Initial value matches a's shape/type but zeroed
    init_val = jnp.zeros_like(a)
    return jax.lax.fori_loop(0, n, body_fun, init_val)
'''

def generate_reduction_kernel(seed: int = None) -> str:
    """Generate a reduction kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("reduce")
    
    return f'''@jax.jit
def {name}(a):
    return jnp.sum(a)
'''

def generate_vector_kernel(seed: int = None) -> str:
    """Generate a kernel with vector operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("vec")
    ops = ["jnp.dot", "jnp.linalg.norm"]
    op = random.choice(ops)
    
    if op == "jnp.dot":
        # Assumes inputs are vectors (1D arrays) or batches of vectors
        # For simple mapping to Warp vec3, we assume last dim is 3.
        # But here we just write generic JAX.
        return f'''@jax.jit
def {name}(a, b):
    # vector dot product along last axis
    return jnp.sum(a * b, axis=-1)
'''
    else:
        return f'''@jax.jit
def {name}(a):
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
    
    return f'''@jax.jit
def {name}(a, b):
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
    
    return f'''@jax.jit
def {name}(a):
    # Nested where: if a > t1 then (if a > t2 then ... else ...) else ...
    inner_true = a * 3.0
    inner_false = a * 2.0
    outer_true = jnp.where(a > {t2}, inner_true, inner_false)
    outer_false = a * 0.5
    return jnp.where(a > {t1}, outer_true, outer_false)
'''

def generate_compound_kernel(seed: int = None) -> str:
    """Generate a kernel with compound operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("compound")
    
    return f'''@jax.jit
def {name}(a, b, scale):
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
