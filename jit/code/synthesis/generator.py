"""Kernel Generator - Generates varied JAX functions programmatically."""
import random
import string
from dataclasses import dataclass
from typing import List, Tuple, Callable
import jax
import jax.numpy as jnp


@dataclass
class FunctionSpec:
    """Specification for a generated JAX function."""
    name: str
    source: str
    func: Callable
    sample_inputs: tuple
    input_shapes: List[Tuple]


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

# Reduction operations
REDUCE_OPS = [
    ("jnp.sum", "sum"),
    ("jnp.mean", "mean"),
    ("jnp.max", "max"),
    ("jnp.min", "min"),
]


def random_name(prefix: str = "func") -> str:
    """Generate a random function name."""
    suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f"{prefix}_{suffix}"


def generate_simple_elementwise(seed: int = None) -> FunctionSpec:
    """Generate a simple elementwise function."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("elementwise")
    op_sym, _ = random.choice(BINARY_OPS)
    
    source = f'''def {name}(a, b):
    return a {op_sym} b
'''
    
    # Create the actual function
    exec_globals = {"jnp": jnp}
    exec(source, exec_globals)
    func = exec_globals[name]
    
    # Sample inputs
    n = random.choice([64, 128, 256, 512])
    a = jnp.ones((n,), dtype=jnp.float32)
    b = jnp.ones((n,), dtype=jnp.float32)
    
    return FunctionSpec(
        name=name,
        source=source,
        func=func,
        sample_inputs=(a, b),
        input_shapes=[(n,), (n,)]
    )


def generate_scalar_array_op(seed: int = None) -> FunctionSpec:
    """Generate a function with scalar and array operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("scalar_arr")
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    
    source = f'''def {name}(alpha, x, y):
    return alpha {op1_sym} x {op2_sym} y
'''
    
    exec_globals = {"jnp": jnp}
    exec(source, exec_globals)
    func = exec_globals[name]
    
    n = random.choice([64, 128, 256])
    alpha = jnp.array(2.0, dtype=jnp.float32)
    x = jnp.ones((n,), dtype=jnp.float32)
    y = jnp.ones((n,), dtype=jnp.float32)
    
    return FunctionSpec(
        name=name,
        source=source,
        func=func,
        sample_inputs=(alpha, x, y),
        input_shapes=[(), (n,), (n,)]
    )


def generate_unary_function(seed: int = None) -> FunctionSpec:
    """Generate a function with unary operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("unary")
    op_func, _ = random.choice(UNARY_OPS)
    
    source = f'''def {name}(a):
    return {op_func}(a)
'''
    
    exec_globals = {"jnp": jnp}
    exec(source, exec_globals)
    func = exec_globals[name]
    
    n = random.choice([64, 128, 256])
    # Use positive values for sqrt/log safety
    a = jnp.ones((n,), dtype=jnp.float32) + 0.5
    
    return FunctionSpec(
        name=name,
        source=source,
        func=func,
        sample_inputs=(a,),
        input_shapes=[(n,)]
    )


def generate_branch_function(seed: int = None) -> FunctionSpec:
    """Generate a function with branching using jnp.where."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("branch")
    threshold = round(random.uniform(-1.0, 1.0), 2)
    op1_sym, _ = random.choice(BINARY_OPS)
    const1 = round(random.uniform(0.5, 3.0), 1)
    const2 = round(random.uniform(0.5, 3.0), 1)
    
    source = f'''def {name}(a):
    return jnp.where(a > {threshold}, a {op1_sym} {const1}, a * {const2})
'''
    
    exec_globals = {"jnp": jnp}
    exec(source, exec_globals)
    func = exec_globals[name]
    
    n = random.choice([64, 128, 256])
    a = jnp.linspace(-2.0, 2.0, n, dtype=jnp.float32)
    
    return FunctionSpec(
        name=name,
        source=source,
        func=func,
        sample_inputs=(a,),
        input_shapes=[(n,)]
    )


def generate_reduction_function(seed: int = None) -> FunctionSpec:
    """Generate a reduction function."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("reduce")
    op_func, _ = random.choice(REDUCE_OPS)
    
    source = f'''def {name}(a):
    return {op_func}(a)
'''
    
    exec_globals = {"jnp": jnp}
    exec(source, exec_globals)
    func = exec_globals[name]
    
    n = random.choice([64, 128, 256])
    a = jnp.ones((n,), dtype=jnp.float32)
    
    return FunctionSpec(
        name=name,
        source=source,
        func=func,
        sample_inputs=(a,),
        input_shapes=[(n,)]
    )


def generate_dot_product(seed: int = None) -> FunctionSpec:
    """Generate a dot product function."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("dot")
    
    source = f'''def {name}(a, b):
    return jnp.dot(a, b)
'''
    
    exec_globals = {"jnp": jnp}
    exec(source, exec_globals)
    func = exec_globals[name]
    
    n = random.choice([64, 128, 256])
    a = jnp.ones((n,), dtype=jnp.float32)
    b = jnp.ones((n,), dtype=jnp.float32)
    
    return FunctionSpec(
        name=name,
        source=source,
        func=func,
        sample_inputs=(a, b),
        input_shapes=[(n,), (n,)]
    )


def generate_matmul_function(seed: int = None) -> FunctionSpec:
    """Generate a matrix multiplication function."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("matmul")
    
    source = f'''def {name}(a, b):
    return jnp.matmul(a, b)
'''
    
    exec_globals = {"jnp": jnp}
    exec(source, exec_globals)
    func = exec_globals[name]
    
    m = random.choice([32, 64, 128])
    n = random.choice([32, 64, 128])
    k = random.choice([32, 64, 128])
    a = jnp.ones((m, k), dtype=jnp.float32)
    b = jnp.ones((k, n), dtype=jnp.float32)
    
    return FunctionSpec(
        name=name,
        source=source,
        func=func,
        sample_inputs=(a, b),
        input_shapes=[(m, k), (k, n)]
    )


def generate_multi_statement_function(seed: int = None) -> FunctionSpec:
    """Generate a function with multiple statements."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("multi")
    op1_sym, _ = random.choice(BINARY_OPS)
    op2_sym, _ = random.choice(BINARY_OPS)
    unary_op, _ = random.choice(UNARY_OPS)
    
    source = f'''def {name}(a, b):
    temp1 = a {op1_sym} b
    temp2 = {unary_op}(temp1)
    return temp2 {op2_sym} a
'''
    
    exec_globals = {"jnp": jnp}
    exec(source, exec_globals)
    func = exec_globals[name]
    
    n = random.choice([64, 128, 256])
    a = jnp.ones((n,), dtype=jnp.float32) + 0.5
    b = jnp.ones((n,), dtype=jnp.float32) + 0.5
    
    return FunctionSpec(
        name=name,
        source=source,
        func=func,
        sample_inputs=(a, b),
        input_shapes=[(n,), (n,)]
    )


def generate_nested_branch_function(seed: int = None) -> FunctionSpec:
    """Generate a function with nested conditionals."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("nested")
    t1 = round(random.uniform(-0.5, 0.5), 2)
    t2 = round(random.uniform(0.5, 1.5), 2)
    
    source = f'''def {name}(a):
    cond1 = a > {t1}
    cond2 = a > {t2}
    inner = jnp.where(cond2, a * 3.0, a * 2.0)
    return jnp.where(cond1, inner, a * 0.5)
'''
    
    exec_globals = {"jnp": jnp}
    exec(source, exec_globals)
    func = exec_globals[name]
    
    n = random.choice([64, 128, 256])
    a = jnp.linspace(-2.0, 2.0, n, dtype=jnp.float32)
    
    return FunctionSpec(
        name=name,
        source=source,
        func=func,
        sample_inputs=(a,),
        input_shapes=[(n,)]
    )


def generate_compound_function(seed: int = None) -> FunctionSpec:
    """Generate a compound function with multiple operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("compound")
    
    source = f'''def {name}(a, b, scale):
    x = a
    y = b
    result = (x + y) * scale
    result = result - jnp.floor(result)
    return jnp.abs(result)
'''
    
    exec_globals = {"jnp": jnp}
    exec(source, exec_globals)
    func = exec_globals[name]
    
    n = random.choice([64, 128, 256])
    a = jnp.ones((n,), dtype=jnp.float32)
    b = jnp.ones((n,), dtype=jnp.float32)
    scale = jnp.array(2.5, dtype=jnp.float32)
    
    return FunctionSpec(
        name=name,
        source=source,
        func=func,
        sample_inputs=(a, b, scale),
        input_shapes=[(n,), (n,), ()]
    )


def generate_softmax_function(seed: int = None) -> FunctionSpec:
    """Generate a softmax function."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("softmax")
    axis = random.choice([-1, 0])
    
    source = f'''def {name}(x):
    x_max = jnp.max(x, axis={axis}, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis={axis}, keepdims=True)
'''
    
    exec_globals = {"jnp": jnp}
    exec(source, exec_globals)
    func = exec_globals[name]
    
    n = random.choice([64, 128, 256])
    x = jnp.ones((n,), dtype=jnp.float32)
    
    return FunctionSpec(
        name=name,
        source=source,
        func=func,
        sample_inputs=(x,),
        input_shapes=[(n,)]
    )


# All generator functions
GENERATORS = [
    generate_simple_elementwise,
    generate_scalar_array_op,
    generate_unary_function,
    generate_branch_function,
    generate_reduction_function,
    generate_dot_product,
    generate_matmul_function,
    generate_multi_statement_function,
    generate_nested_branch_function,
    generate_compound_function,
]


def generate_random_function(seed: int = None) -> FunctionSpec:
    """Generate a random function from available templates."""
    if seed is not None:
        random.seed(seed)
    generator = random.choice(GENERATORS)
    return generator(seed)


def generate_function_batch(count: int, base_seed: int = 42) -> List[FunctionSpec]:
    """Generate a batch of unique functions."""
    functions = []
    for i in range(count):
        func_spec = generate_random_function(base_seed + i)
        functions.append(func_spec)
    return functions


if __name__ == "__main__":
    # Test function generation
    print("=== Testing JAX Function Generator ===\n")
    
    for i, gen_func in enumerate(GENERATORS):
        print(f"--- Generator: {gen_func.__name__} ---")
        func_spec = gen_func(seed=42 + i)
        print(f"Name: {func_spec.name}")
        print(f"Source:\n{func_spec.source}")
        
        # Test the function
        result = func_spec.func(*func_spec.sample_inputs)
        print(f"Output shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")
        print()
    
    print("\n=== Batch Generation Test ===")
    batch = generate_function_batch(5, base_seed=100)
    for i, spec in enumerate(batch):
        print(f"Function {i+1}: {spec.name}")
        print(spec.source)
