"""Kernel Generator - Generates varied JAX functions programmatically."""
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
    ("jnp.log", "log"),
    ("jnp.tanh", "tanh"),
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
    """Scalar-array combined operation."""
    return alpha {op1_sym} x {op2_sym} y
'''


def generate_unary_kernel(seed: int = None) -> str:
    """Generate a kernel with unary operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("unary")
    op_func, _ = random.choice(UNARY_OPS)
    
    return f'''def {name}(a):
    """Unary math operation."""
    return {op_func}(a)
'''


def generate_branch_kernel(seed: int = None) -> str:
    """Generate a kernel with branching using jax.lax.cond."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("branch")
    threshold = round(random.uniform(-1.0, 1.0), 2)
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    const1 = round(random.uniform(0.5, 3.0), 1)
    const2 = round(random.uniform(0.5, 3.0), 1)
    
    return f'''def {name}(a):
    """Conditional operation with threshold."""
    return jnp.where(a > {threshold}, a {op1_sym} {const1}, a {op2_sym} {const2})
'''


def generate_loop_kernel(seed: int = None) -> str:
    """Generate a kernel with a scan/loop pattern."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("loop")
    op_sym, _ = random.choice(BINARY_OPS)
    n_iters = random.randint(2, 5)
    
    return f'''def {name}(a, n):
    """Loop-based accumulation using lax.fori_loop."""
    def body_fn(i, acc):
        return acc {op_sym} a
    return jax.lax.fori_loop(0, n, body_fn, jnp.zeros_like(a))
'''


def generate_reduction_kernel(seed: int = None) -> str:
    """Generate a reduction kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("reduce")
    ops = ["jnp.sum", "jnp.prod", "jnp.mean", "jnp.max", "jnp.min"]
    op = random.choice(ops)
    
    return f'''def {name}(a):
    """Reduction operation over array."""
    return {op}(a)
'''


def generate_vector_kernel(seed: int = None) -> str:
    """Generate a kernel with vector operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("vec")
    ops = ["dot", "norm", "cross"]
    op = random.choice(ops[:2])  # dot or norm
    
    if op == "dot":
        return f'''def {name}(a, b):
    """Vector dot product."""
    return jnp.sum(a * b, axis=-1)
'''
    else:
        return f'''def {name}(a):
    """Vector norm computation."""
    return jnp.sqrt(jnp.sum(a * a, axis=-1))
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
    """Multi-statement computation chain."""
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
    """Nested conditional logic."""
    cond1 = a > {t1}
    cond2 = a > {t2}
    result = jnp.where(cond1,
                       jnp.where(cond2, a * 3.0, a * 2.0),
                       a * 0.5)
    return result
'''


def generate_compound_kernel(seed: int = None) -> str:
    """Generate a kernel with compound operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("compound")
    
    return f'''def {name}(a, b, scale):
    """Compound multi-operation computation."""
    x = a
    y = b
    result = (x + y) * scale
    result = result - jnp.floor(result)
    return jnp.abs(result)
'''


def generate_matmul_kernel(seed: int = None) -> str:
    """Generate a matrix multiplication kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("matmul")
    
    return f'''def {name}(a, b):
    """Matrix multiplication."""
    return jnp.matmul(a, b)
'''


def generate_softmax_kernel(seed: int = None) -> str:
    """Generate a softmax kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("softmax")
    axis = random.choice([-1, 0, 1])
    
    return f'''def {name}(x):
    """Softmax activation along axis."""
    x_max = jnp.max(x, axis={axis}, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis={axis}, keepdims=True)
'''


def generate_attention_kernel(seed: int = None) -> str:
    """Generate an attention-like kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("attention")
    
    return f'''def {name}(q, k, v):
    """Scaled dot-product attention."""
    d_k = q.shape[-1]
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, v)
'''


def generate_conv_kernel(seed: int = None) -> str:
    """Generate a simple convolution-like kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("conv")
    
    return f'''def {name}(x, kernel):
    """1D convolution-like operation."""
    return jax.lax.conv_general_dilated(
        x[None, :, None],
        kernel[None, :, None],
        window_strides=(1,),
        padding='SAME'
    )[0, :, 0]
'''


def generate_layernorm_kernel(seed: int = None) -> str:
    """Generate a layer normalization kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("layernorm")
    eps = random.choice([1e-5, 1e-6, 1e-8])
    
    return f'''def {name}(x, gamma, beta):
    """Layer normalization."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + {eps}) + beta
'''


def generate_gelu_kernel(seed: int = None) -> str:
    """Generate a GELU activation kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("gelu")
    
    return f'''def {name}(x):
    """GELU activation function."""
    return x * 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))
'''


def generate_dropout_kernel(seed: int = None) -> str:
    """Generate a dropout kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("dropout")
    rate = round(random.uniform(0.1, 0.5), 2)
    
    return f'''def {name}(x, key, training=True):
    """Dropout regularization."""
    if not training:
        return x
    keep_rate = {1.0 - rate}
    mask = jax.random.bernoulli(key, keep_rate, x.shape)
    return jnp.where(mask, x / keep_rate, 0.0)
'''


def generate_batch_norm_kernel(seed: int = None) -> str:
    """Generate a batch normalization kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("batchnorm")
    eps = random.choice([1e-5, 1e-3])
    
    return f'''def {name}(x, gamma, beta):
    """Batch normalization over first axis."""
    mean = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.var(x, axis=0, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + {eps})
    return gamma * x_norm + beta
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

# Extended generators for ML-focused kernels
EXTENDED_GENERATORS = GENERATORS + [
    generate_matmul_kernel,
    generate_softmax_kernel,
    generate_attention_kernel,
    generate_layernorm_kernel,
    generate_gelu_kernel,
    generate_batch_norm_kernel,
]


def generate_random_kernel(seed: int = None, extended: bool = False) -> str:
    """Generate a random kernel from available templates."""
    if seed is not None:
        random.seed(seed)
    generators = EXTENDED_GENERATORS if extended else GENERATORS
    generator = random.choice(generators)
    return generator(seed)


def generate_kernel_batch(count: int, base_seed: int = 42, extended: bool = False) -> List[str]:
    """Generate a batch of unique kernels."""
    kernels = []
    for i in range(count):
        kernel = generate_random_kernel(base_seed + i, extended)
        kernels.append(kernel)
    return kernels


if __name__ == "__main__":
    # Test kernel generation
    print("=== Testing JAX Kernel Generator ===\n")
    
    for i, gen_func in enumerate(GENERATORS):
        print(f"--- Generator: {gen_func.__name__} ---")
        kernel_src = gen_func(seed=42 + i)
        print(kernel_src)
    
    print("\n=== Extended Generators ===")
    for gen_func in EXTENDED_GENERATORS[10:]:
        print(f"--- Generator: {gen_func.__name__} ---")
        kernel_src = gen_func(seed=42)
        print(kernel_src)
    
    print("\n=== Batch Generation Test ===")
    batch = generate_kernel_batch(5, base_seed=100)
    for i, k in enumerate(batch):
        print(f"Kernel {i+1}:")
        print(k)
