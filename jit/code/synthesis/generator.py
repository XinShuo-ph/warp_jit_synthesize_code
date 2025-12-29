"""Kernel Generator - Generates varied Warp kernels programmatically."""
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


# Type definitions for kernel arguments
ARRAY_TYPES = [
    "wp.array(dtype=float)",
    "wp.array(dtype=int)",
    "wp.array(dtype=wp.vec3)",
    "wp.array(dtype=wp.vec2)",
]

SCALAR_TYPES = ["float", "int"]

# Binary operations
BINARY_OPS = [
    ("+", "wp.add"),
    ("-", "wp.sub"),
    ("*", "wp.mul"),
]

# Unary operations
UNARY_OPS = [
    ("wp.sqrt", "sqrt"),
    ("wp.abs", "abs"),
    ("wp.sin", "sin"),
    ("wp.cos", "cos"),
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
    
    return f'''@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] {op_sym} b[tid]
'''


def generate_scalar_array_op(seed: int = None) -> str:
    """Generate a kernel with scalar and array operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("scalar_arr")
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    
    return f'''@wp.kernel
def {name}(alpha: float, x: wp.array(dtype=float), y: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = alpha {op1_sym} x[tid] {op2_sym} y[tid]
'''


def generate_unary_kernel(seed: int = None) -> str:
    """Generate a kernel with unary operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("unary")
    op_func, _ = random.choice(UNARY_OPS)
    
    return f'''@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = {op_func}(a[tid])
'''


def generate_branch_kernel(seed: int = None) -> str:
    """Generate a kernel with branching."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("branch")
    threshold = round(random.uniform(-1.0, 1.0), 2)
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    const1 = round(random.uniform(0.5, 3.0), 1)
    const2 = round(random.uniform(0.5, 3.0), 1)
    
    return f'''@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    val = a[tid]
    if val > {threshold}:
        b[tid] = val {op1_sym} {const1}
    else:
        b[tid] = val {op2_sym} {const2}
'''


def generate_loop_kernel(seed: int = None) -> str:
    """Generate a kernel with a loop."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("loop")
    op_sym, _ = random.choice(BINARY_OPS)
    
    return f'''@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float), n: int):
    tid = wp.tid()
    total = float(0.0)
    for i in range(n):
        total = total {op_sym} a[tid]
    b[tid] = total
'''


def generate_reduction_kernel(seed: int = None) -> str:
    """Generate an atomic reduction kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("reduce")
    
    return f'''@wp.kernel
def {name}(a: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(result, 0, a[tid])
'''


def generate_vector_kernel(seed: int = None) -> str:
    """Generate a kernel with vector operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("vec")
    ops = ["wp.dot", "wp.length", "wp.cross"]
    op = random.choice(ops[:2])  # dot or length
    
    if op == "wp.dot":
        return f'''@wp.kernel
def {name}(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = wp.dot(a[tid], b[tid])
'''
    else:
        return f'''@wp.kernel
def {name}(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = wp.length(a[tid])
'''


def generate_multi_statement_kernel(seed: int = None) -> str:
    """Generate a kernel with multiple statements."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("multi")
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    unary_op, _ = random.choice(UNARY_OPS)
    
    return f'''@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    temp1 = a[tid] {op1_sym} b[tid]
    temp2 = {unary_op}(temp1)
    c[tid] = temp2 {op2_sym} a[tid]
'''


def generate_nested_branch_kernel(seed: int = None) -> str:
    """Generate a kernel with nested branches."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("nested")
    t1 = round(random.uniform(-0.5, 0.5), 2)
    t2 = round(random.uniform(0.5, 1.5), 2)
    
    return f'''@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    val = a[tid]
    if val > {t1}:
        if val > {t2}:
            b[tid] = val * 3.0
        else:
            b[tid] = val * 2.0
    else:
        b[tid] = val * 0.5
'''


def generate_compound_kernel(seed: int = None) -> str:
    """Generate a kernel with compound operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("compound")
    
    return f'''@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float), scale: float):
    tid = wp.tid()
    x = a[tid]
    y = b[tid]
    result = (x + y) * scale
    result = result - wp.floor(result)
    c[tid] = wp.abs(result)
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
