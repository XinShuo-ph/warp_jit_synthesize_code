"""Generate varied Python/JAX functions programmatically."""

import random
import jax.numpy as jnp
import jax
from jax import lax


# Operation templates
UNARY_OPS = [
    ("jnp.sin", "sin"),
    ("jnp.cos", "cos"),
    ("jnp.exp", "exp"),
    ("jnp.log", "log"),
    ("jnp.abs", "abs"),
    ("jnp.sqrt", "sqrt"),
    ("jnp.tanh", "tanh"),
    ("jnp.negative", "negative"),
    ("jnp.square", "square"),
]

BINARY_OPS = [
    ("+", "add"),
    ("-", "sub"),
    ("*", "mul"),
    ("/", "div"),
    ("**", "pow"),
    ("jnp.maximum", "max"),
    ("jnp.minimum", "min"),
]

REDUCE_OPS = [
    ("jnp.sum", "sum"),
    ("jnp.mean", "mean"),
    ("jnp.max", "max"),
    ("jnp.min", "min"),
    ("jnp.prod", "prod"),
]

SHAPES = [
    (4,),
    (8,),
    (16,),
    (4, 4),
    (8, 8),
    (4, 8),
    (8, 4),
    (2, 4, 4),
]


def gen_simple_unary():
    """Generate simple unary operation."""
    op_str, op_name = random.choice(UNARY_OPS)
    shape = random.choice(SHAPES[:4])  # 1D shapes
    
    source = f"def f(x): return {op_str}(x)"
    
    if op_name in ["log", "sqrt"]:
        def fn(x): 
            return getattr(jnp, op_name)(jnp.abs(x) + 0.1)
        source = f"def f(x): return {op_str}(jnp.abs(x) + 0.1)"
    else:
        def fn(x):
            return getattr(jnp, op_name)(x)
    
    example_input = jnp.ones(shape)
    return fn, source, (example_input,)


def gen_binary_chain():
    """Generate chain of binary operations."""
    num_ops = random.randint(2, 4)
    ops = [random.choice(BINARY_OPS) for _ in range(num_ops)]
    shape = random.choice(SHAPES[:4])
    
    # Build source
    expr = "x"
    for i, (op_str, _) in enumerate(ops):
        if op_str.startswith("jnp."):
            expr = f"{op_str}({expr}, y)"
        else:
            expr = f"({expr} {op_str} y)"
    source = f"def f(x, y): return {expr}"
    
    # Build function
    def make_fn(operations):
        def fn(x, y):
            result = x
            for op_str, op_name in operations:
                if op_name == "add":
                    result = result + y
                elif op_name == "sub":
                    result = result - y
                elif op_name == "mul":
                    result = result * y
                elif op_name == "div":
                    result = result / (y + 0.1)
                elif op_name == "pow":
                    result = result ** 2
                elif op_name == "max":
                    result = jnp.maximum(result, y)
                elif op_name == "min":
                    result = jnp.minimum(result, y)
            return result
        return fn
    
    fn = make_fn(ops)
    x = jnp.ones(shape)
    y = jnp.ones(shape) * 0.5
    return fn, source, (x, y)


def gen_reduce_op():
    """Generate reduction operation."""
    op_str, op_name = random.choice(REDUCE_OPS)
    shape = random.choice(SHAPES[4:])  # 2D+ shapes
    axis = random.choice([None, 0, -1])
    
    if axis is None:
        source = f"def f(x): return {op_str}(x)"
    else:
        source = f"def f(x): return {op_str}(x, axis={axis})"
    
    def fn(x):
        return getattr(jnp, op_name)(x, axis=axis)
    
    example_input = jnp.ones(shape)
    return fn, source, (example_input,)


def gen_matmul():
    """Generate matrix multiplication."""
    m, k, n = random.choice([(4, 4, 4), (8, 4, 8), (4, 8, 4), (16, 8, 16)])
    
    source = f"def f(a, b): return jnp.dot(a, b)"
    
    def fn(a, b):
        return jnp.dot(a, b)
    
    a = jnp.ones((m, k))
    b = jnp.ones((k, n))
    return fn, source, (a, b)


def gen_activation():
    """Generate activation function patterns."""
    act_type = random.choice(["relu", "sigmoid", "softmax", "gelu", "leaky_relu"])
    shape = random.choice(SHAPES[:4])
    
    if act_type == "relu":
        source = "def f(x): return jnp.maximum(x, 0)"
        def fn(x): return jnp.maximum(x, 0)
    elif act_type == "sigmoid":
        source = "def f(x): return 1 / (1 + jnp.exp(-x))"
        def fn(x): return 1 / (1 + jnp.exp(-x))
    elif act_type == "softmax":
        source = "def f(x): e = jnp.exp(x - jnp.max(x)); return e / jnp.sum(e)"
        def fn(x): 
            e = jnp.exp(x - jnp.max(x))
            return e / jnp.sum(e)
    elif act_type == "gelu":
        source = "def f(x): return x * 0.5 * (1 + jnp.tanh(0.7978845608 * (x + 0.044715 * x**3)))"
        def fn(x): return x * 0.5 * (1 + jnp.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    else:  # leaky_relu
        source = "def f(x): return jnp.where(x > 0, x, 0.01 * x)"
        def fn(x): return jnp.where(x > 0, x, 0.01 * x)
    
    example_input = jnp.ones(shape)
    return fn, source, (example_input,)


def gen_normalize():
    """Generate normalization patterns."""
    norm_type = random.choice(["layer_norm", "batch_norm", "l2_norm"])
    shape = random.choice(SHAPES[4:6])  # 2D shapes
    
    if norm_type == "layer_norm":
        source = "def f(x): m = jnp.mean(x, axis=-1, keepdims=True); s = jnp.std(x, axis=-1, keepdims=True); return (x - m) / (s + 1e-5)"
        def fn(x):
            m = jnp.mean(x, axis=-1, keepdims=True)
            s = jnp.std(x, axis=-1, keepdims=True)
            return (x - m) / (s + 1e-5)
    elif norm_type == "batch_norm":
        source = "def f(x): m = jnp.mean(x, axis=0, keepdims=True); s = jnp.std(x, axis=0, keepdims=True); return (x - m) / (s + 1e-5)"
        def fn(x):
            m = jnp.mean(x, axis=0, keepdims=True)
            s = jnp.std(x, axis=0, keepdims=True)
            return (x - m) / (s + 1e-5)
    else:  # l2_norm
        source = "def f(x): return x / jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1e-5)"
        def fn(x):
            return x / jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1e-5)
    
    example_input = jnp.ones(shape)
    return fn, source, (example_input,)


def gen_vmap_pattern():
    """Generate vmap pattern."""
    batch_size = random.choice([4, 8, 16])
    inner_shape = random.choice([(4,), (8,), (4, 4)])
    
    source = f"def f(x, y): return jax.vmap(lambda a, b: jnp.dot(a, b))(x, y)"
    
    def fn(x, y):
        return jax.vmap(lambda a, b: jnp.dot(a, b))(x, y)
    
    if len(inner_shape) == 1:
        x = jnp.ones((batch_size,) + inner_shape)
        y = jnp.ones((batch_size,) + inner_shape)
    else:
        x = jnp.ones((batch_size,) + inner_shape)
        y = jnp.ones((batch_size, inner_shape[1], inner_shape[0]))
    
    return fn, source, (x, y)


def gen_scan_pattern():
    """Generate scan pattern."""
    seq_len = random.choice([4, 8, 16])
    hidden_size = random.choice([4, 8])
    
    source = f"def f(init, xs): return lax.scan(lambda c, x: (c * 0.9 + x * 0.1, c), init, xs)"
    
    def fn(init, xs):
        return lax.scan(lambda c, x: (c * 0.9 + x * 0.1, c), init, xs)
    
    init = jnp.ones(hidden_size)
    xs = jnp.ones((seq_len, hidden_size))
    return fn, source, (init, xs)


GENERATORS = [
    gen_simple_unary,
    gen_binary_chain,
    gen_reduce_op,
    gen_matmul,
    gen_activation,
    gen_normalize,
    gen_vmap_pattern,
    gen_scan_pattern,
]


def generate_random_function():
    """Generate a random function with its source and example inputs."""
    generator = random.choice(GENERATORS)
    return generator()


if __name__ == "__main__":
    # Test generators
    for gen in GENERATORS:
        fn, source, args = gen()
        try:
            jaxpr = jax.make_jaxpr(fn)(*args)
            print(f"{gen.__name__}: OK")
            print(f"  Source: {source[:60]}...")
        except Exception as e:
            print(f"{gen.__name__}: FAILED - {e}")
    
    print("\n=== All generators tested ===")
