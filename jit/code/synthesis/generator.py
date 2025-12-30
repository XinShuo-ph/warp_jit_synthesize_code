"""Kernel Generator - Generates varied JAX functions programmatically."""
import random
import string
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class KernelSpec:
    """Specification for a function to generate."""
    name: str
    args: List[Tuple[str, str]]  # [(name, type_hint), ...]
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
    ("jnp.exp", "exp"),
    ("jnp.log1p", "log1p"),
    ("jnp.tanh", "tanh"),
]


def random_name(prefix: str = "kernel") -> str:
    """Generate a random kernel name."""
    suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f"{prefix}_{suffix}"


def generate_simple_elementwise(seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a simple elementwise function."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("elementwise")
    op_sym, _ = random.choice(BINARY_OPS)
    
    source = f'''def {name}(a, b):
    """Elementwise operation on two arrays."""
    return a {op_sym} b
'''
    
    # Sample args generator
    sample_args_code = '''
import jax.numpy as jnp
import jax
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (100,))
b = jax.random.normal(jax.random.PRNGKey(43), (100,))
'''
    
    return source, {"sample_args_code": sample_args_code, "arg_names": ["a", "b"]}


def generate_scalar_array_op(seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a function with scalar and array operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("scalar_arr")
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    
    source = f'''def {name}(alpha, x, y):
    """Scalar-array operation: alpha op1 x op2 y."""
    return alpha {op1_sym} x {op2_sym} y
'''
    
    sample_args_code = '''
import jax.numpy as jnp
import jax
key = jax.random.PRNGKey(42)
alpha = 2.5
x = jax.random.normal(key, (100,))
y = jax.random.normal(jax.random.PRNGKey(43), (100,))
'''
    
    return source, {"sample_args_code": sample_args_code, "arg_names": ["alpha", "x", "y"]}


def generate_unary_kernel(seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a function with unary operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("unary")
    op_func, _ = random.choice(UNARY_OPS)
    
    source = f'''def {name}(a):
    """Unary operation on array."""
    return {op_func}(jnp.abs(a) + 0.1)  # Ensure positive for sqrt/log
'''
    
    sample_args_code = '''
import jax.numpy as jnp
import jax
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (100,))
'''
    
    return source, {"sample_args_code": sample_args_code, "arg_names": ["a"]}


def generate_branch_kernel(seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a function with branching (using jnp.where)."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("branch")
    threshold = round(random.uniform(-1.0, 1.0), 2)
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    const1 = round(random.uniform(0.5, 3.0), 1)
    const2 = round(random.uniform(0.5, 3.0), 1)
    
    source = f'''def {name}(a):
    """Conditional operation using jnp.where."""
    return jnp.where(a > {threshold}, a {op1_sym} {const1}, a {op2_sym} {const2})
'''
    
    sample_args_code = '''
import jax.numpy as jnp
import jax
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (100,))
'''
    
    return source, {"sample_args_code": sample_args_code, "arg_names": ["a"]}


def generate_loop_kernel(seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a function with a loop (using jax.lax.fori_loop)."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("loop")
    op_sym, _ = random.choice(BINARY_OPS)
    n_iters = random.randint(3, 10)
    
    source = f'''def {name}(a):
    """Loop operation using jax.lax.fori_loop."""
    def body_fn(i, total):
        return total {op_sym} a
    return jax.lax.fori_loop(0, {n_iters}, body_fn, jnp.zeros_like(a))
'''
    
    sample_args_code = '''
import jax.numpy as jnp
import jax
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (100,))
'''
    
    return source, {"sample_args_code": sample_args_code, "arg_names": ["a"]}


def generate_reduction_kernel(seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a reduction function."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("reduce")
    reduction_ops = ["jnp.sum", "jnp.mean", "jnp.max", "jnp.min", "jnp.prod"]
    op = random.choice(reduction_ops[:3])  # sum, mean, or max
    
    source = f'''def {name}(a):
    """Reduction operation."""
    return {op}(a)
'''
    
    sample_args_code = '''
import jax.numpy as jnp
import jax
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (100,))
'''
    
    return source, {"sample_args_code": sample_args_code, "arg_names": ["a"]}


def generate_vector_kernel(seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a function with vector/matrix operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("vec")
    ops = ["dot", "norm", "cross"]
    op = random.choice(ops[:2])  # dot or norm
    
    if op == "dot":
        source = f'''def {name}(a, b):
    """Dot product operation."""
    return jnp.sum(a * b, axis=-1)
'''
        sample_args_code = '''
import jax.numpy as jnp
import jax
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (100, 3))
b = jax.random.normal(jax.random.PRNGKey(43), (100, 3))
'''
        arg_names = ["a", "b"]
    else:
        source = f'''def {name}(a):
    """Vector norm operation."""
    return jnp.sqrt(jnp.sum(a ** 2, axis=-1))
'''
        sample_args_code = '''
import jax.numpy as jnp
import jax
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (100, 3))
'''
        arg_names = ["a"]
    
    return source, {"sample_args_code": sample_args_code, "arg_names": arg_names}


def generate_multi_statement_kernel(seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a function with multiple statements."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("multi")
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    unary_op, _ = random.choice(UNARY_OPS)
    
    source = f'''def {name}(a, b):
    """Multi-statement operation."""
    temp1 = a {op1_sym} b
    temp2 = {unary_op}(jnp.abs(temp1) + 0.1)
    return temp2 {op2_sym} a
'''
    
    sample_args_code = '''
import jax.numpy as jnp
import jax
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (100,))
b = jax.random.normal(jax.random.PRNGKey(43), (100,))
'''
    
    return source, {"sample_args_code": sample_args_code, "arg_names": ["a", "b"]}


def generate_nested_branch_kernel(seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a function with nested conditionals."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("nested")
    t1 = round(random.uniform(-0.5, 0.5), 2)
    t2 = round(random.uniform(0.5, 1.5), 2)
    
    source = f'''def {name}(a):
    """Nested conditional operation."""
    inner = jnp.where(a > {t2}, a * 3.0, a * 2.0)
    return jnp.where(a > {t1}, inner, a * 0.5)
'''
    
    sample_args_code = '''
import jax.numpy as jnp
import jax
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (100,))
'''
    
    return source, {"sample_args_code": sample_args_code, "arg_names": ["a"]}


def generate_compound_kernel(seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a function with compound operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("compound")
    
    source = f'''def {name}(a, b, scale):
    """Compound operation with multiple steps."""
    x = a
    y = b
    result = (x + y) * scale
    result = result - jnp.floor(result)
    return jnp.abs(result)
'''
    
    sample_args_code = '''
import jax.numpy as jnp
import jax
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (100,))
b = jax.random.normal(jax.random.PRNGKey(43), (100,))
scale = 2.0
'''
    
    return source, {"sample_args_code": sample_args_code, "arg_names": ["a", "b", "scale"]}


def generate_matmul_kernel(seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a matrix multiplication function."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("matmul")
    
    source = f'''def {name}(a, b):
    """Matrix multiplication."""
    return jnp.matmul(a, b)
'''
    
    sample_args_code = '''
import jax.numpy as jnp
import jax
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (32, 64))
b = jax.random.normal(jax.random.PRNGKey(43), (64, 32))
'''
    
    return source, {"sample_args_code": sample_args_code, "arg_names": ["a", "b"]}


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
]


def generate_random_kernel(seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a random function from available templates."""
    if seed is not None:
        random.seed(seed)
    generator = random.choice(GENERATORS)
    return generator(seed)


def generate_kernel_batch(count: int, base_seed: int = 42) -> List[Tuple[str, Dict[str, Any]]]:
    """Generate a batch of unique functions."""
    kernels = []
    for i in range(count):
        kernel = generate_random_kernel(base_seed + i)
        kernels.append(kernel)
    return kernels


if __name__ == "__main__":
    # Test kernel generation
    print("=== Testing JAX Function Generator ===\n")
    
    for i, gen_func in enumerate(GENERATORS):
        print(f"--- Generator: {gen_func.__name__} ---")
        kernel_src, metadata = gen_func(seed=42 + i)
        print(kernel_src)
        print(f"Args: {metadata['arg_names']}")
        print()
    
    print("\n=== Batch Generation Test ===")
    batch = generate_kernel_batch(5, base_seed=100)
    for i, (k, meta) in enumerate(batch):
        print(f"Function {i+1}:")
        print(k)
