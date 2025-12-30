"""Function generator for JAX synthesis pipeline.

Generates diverse Python functions that can be JIT-compiled with JAX.
"""
import random
import string
from typing import List, Tuple
import jax.numpy as jnp


# Operation templates
UNARY_OPS = [
    ("sin", "jnp.sin({x})"),
    ("cos", "jnp.cos({x})"),
    ("tan", "jnp.tan({x})"),
    ("exp", "jnp.exp({x})"),
    ("log", "jnp.log(jnp.abs({x}) + 1e-6)"),
    ("sqrt", "jnp.sqrt(jnp.abs({x}))"),
    ("abs", "jnp.abs({x})"),
    ("neg", "-{x}"),
    ("square", "{x} ** 2"),
    ("tanh", "jnp.tanh({x})"),
    ("sigmoid", "1 / (1 + jnp.exp(-{x}))"),
    ("relu", "jnp.maximum({x}, 0)"),
]

BINARY_OPS = [
    ("add", "{x} + {y}"),
    ("sub", "{x} - {y}"),
    ("mul", "{x} * {y}"),
    ("div", "{x} / ({y} + 1e-6)"),
    ("max", "jnp.maximum({x}, {y})"),
    ("min", "jnp.minimum({x}, {y})"),
    ("pow", "jnp.power(jnp.abs({x}), jnp.abs({y}) % 3 + 1)"),
]

REDUCTION_OPS = [
    ("sum", "jnp.sum({x})"),
    ("mean", "jnp.mean({x})"),
    ("max", "jnp.max({x})"),
    ("min", "jnp.min({x})"),
    ("prod", "jnp.prod({x})"),
    ("std", "jnp.std({x})"),
]


def random_name(prefix: str = "fn") -> str:
    """Generate a random function name."""
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{suffix}"


def generate_unary_function() -> Tuple[str, str, callable, tuple]:
    """Generate a function with unary operations."""
    name = random_name("unary")
    num_ops = random.randint(1, 3)
    ops = random.sample(UNARY_OPS, min(num_ops, len(UNARY_OPS)))
    
    # Build expression
    expr = "x"
    for _, template in ops:
        expr = template.format(x=expr)
    
    code = f"""def {name}(x):
    return {expr}"""
    
    # Create the function
    namespace = {"jnp": jnp}
    exec(code, namespace)
    fn = namespace[name]
    
    # Example input
    shape = random.choice([(4,), (8,), (2, 4), (3, 3)])
    example_args = (jnp.ones(shape),)
    
    return name, code, fn, example_args


def generate_binary_function() -> Tuple[str, str, callable, tuple]:
    """Generate a function with binary operations."""
    name = random_name("binary")
    num_ops = random.randint(1, 2)
    ops = random.sample(BINARY_OPS, min(num_ops, len(BINARY_OPS)))
    
    # Build expression
    expr = ops[0][1].format(x="x", y="y")
    for _, template in ops[1:]:
        expr = template.format(x=expr, y="y")
    
    code = f"""def {name}(x, y):
    return {expr}"""
    
    namespace = {"jnp": jnp}
    exec(code, namespace)
    fn = namespace[name]
    
    shape = random.choice([(4,), (8,), (2, 4)])
    example_args = (jnp.ones(shape), jnp.ones(shape) * 2)
    
    return name, code, fn, example_args


def generate_reduction_function() -> Tuple[str, str, callable, tuple]:
    """Generate a function with reduction operations."""
    name = random_name("reduce")
    op_name, template = random.choice(REDUCTION_OPS)
    
    # Optionally add pre-processing
    pre_ops = random.sample(UNARY_OPS, random.randint(0, 2))
    pre_expr = "x"
    for _, t in pre_ops:
        pre_expr = t.format(x=pre_expr)
    
    expr = template.format(x=pre_expr)
    
    code = f"""def {name}(x):
    return {expr}"""
    
    namespace = {"jnp": jnp}
    exec(code, namespace)
    fn = namespace[name]
    
    shape = random.choice([(4,), (8,), (16,), (2, 4)])
    example_args = (jnp.ones(shape) + 0.5,)
    
    return name, code, fn, example_args


def generate_mixed_function() -> Tuple[str, str, callable, tuple]:
    """Generate a function with mixed operations."""
    name = random_name("mixed")
    
    # Choose operations
    unary = random.choice(UNARY_OPS)
    binary = random.choice(BINARY_OPS)
    reduction = random.choice(REDUCTION_OPS)
    
    # Build multi-step computation
    lines = [
        f"    t1 = {unary[1].format(x='x')}",
        f"    t2 = {binary[1].format(x='t1', y='y')}",
        f"    t3 = {reduction[1].format(x='t2')}",
        "    return t3"
    ]
    
    code = f"def {name}(x, y):\n" + "\n".join(lines)
    
    namespace = {"jnp": jnp}
    exec(code, namespace)
    fn = namespace[name]
    
    shape = random.choice([(4,), (8,), (2, 4)])
    example_args = (jnp.ones(shape), jnp.ones(shape) * 0.5)
    
    return name, code, fn, example_args


def generate_matmul_function() -> Tuple[str, str, callable, tuple]:
    """Generate a function with matrix operations."""
    name = random_name("matmul")
    
    variants = [
        ("jnp.dot(A, B)", (2, 3), (3, 4)),
        ("jnp.dot(A, B.T)", (2, 3), (4, 3)),
        ("jnp.dot(A.T, B)", (3, 2), (3, 4)),
        ("A @ B", (2, 3), (3, 4)),
    ]
    
    expr, shape_a, shape_b = random.choice(variants)
    
    code = f"""def {name}(A, B):
    return {expr}"""
    
    namespace = {"jnp": jnp}
    exec(code, namespace)
    fn = namespace[name]
    
    example_args = (jnp.ones(shape_a), jnp.ones(shape_b))
    
    return name, code, fn, example_args


def generate_conditional_function() -> Tuple[str, str, callable, tuple]:
    """Generate a function with conditional operations."""
    name = random_name("cond")
    
    conditions = [
        "jnp.where(x > 0, x, 0)",  # ReLU-like
        "jnp.where(x > y, x, y)",  # max-like
        "jnp.where(x > 0.5, jnp.sin(x), jnp.cos(x))",
        "jnp.where(x > jnp.mean(x), x, 0)",
    ]
    
    expr = random.choice(conditions)
    has_y = "y" in expr
    
    if has_y:
        code = f"""def {name}(x, y):
    return {expr}"""
        shape = random.choice([(4,), (8,)])
        example_args = (jnp.array([0.1, 0.6, 0.3, 0.9]), jnp.ones((4,)) * 0.5)
    else:
        code = f"""def {name}(x):
    return {expr}"""
        shape = random.choice([(4,), (8,)])
        example_args = (jnp.array([0.1, 0.6, 0.3, 0.9]),)
    
    namespace = {"jnp": jnp}
    exec(code, namespace)
    fn = namespace[name]
    
    return name, code, fn, example_args


def generate_normalize_function() -> Tuple[str, str, callable, tuple]:
    """Generate normalization functions."""
    name = random_name("norm")
    
    variants = [
        # Standard normalization
        """def {name}(x):
    mean = jnp.mean(x)
    std = jnp.std(x) + 1e-6
    return (x - mean) / std""",
        # L2 normalization
        """def {name}(x):
    norm = jnp.sqrt(jnp.sum(x ** 2)) + 1e-6
    return x / norm""",
        # Min-max normalization
        """def {name}(x):
    min_val = jnp.min(x)
    max_val = jnp.max(x)
    return (x - min_val) / (max_val - min_val + 1e-6)""",
        # Softmax
        """def {name}(x):
    exp_x = jnp.exp(x - jnp.max(x))
    return exp_x / jnp.sum(exp_x)""",
    ]
    
    code = random.choice(variants).format(name=name)
    
    namespace = {"jnp": jnp}
    exec(code, namespace)
    fn = namespace[name]
    
    shape = random.choice([(4,), (8,), (16,)])
    example_args = (jnp.array([1.0, 2.0, 3.0, 4.0]),)
    
    return name, code, fn, example_args


# Generator registry
GENERATORS = [
    generate_unary_function,
    generate_binary_function,
    generate_reduction_function,
    generate_mixed_function,
    generate_matmul_function,
    generate_conditional_function,
    generate_normalize_function,
]


def generate_random_function() -> Tuple[str, str, callable, tuple]:
    """Generate a random function from available generators."""
    generator = random.choice(GENERATORS)
    return generator()


def generate_batch(n: int, seed: int = None) -> List[Tuple[str, str, callable, tuple]]:
    """Generate a batch of n random functions."""
    if seed is not None:
        random.seed(seed)
    
    functions = []
    for _ in range(n):
        try:
            fn_data = generate_random_function()
            functions.append(fn_data)
        except Exception:
            continue  # Skip failed generations
    
    return functions


if __name__ == "__main__":
    # Test generation
    random.seed(42)
    
    print("Generating 10 random functions...\n")
    for i, gen in enumerate(GENERATORS):
        name, code, fn, args = gen()
        print(f"{i+1}. {name}")
        print(code)
        print()
