"""Function generator for creating varied JAX functions programmatically."""
import random
import string
from typing import List, Tuple, Callable
import jax
import jax.numpy as jnp


# =============================================================================
# Templates for different function categories
# =============================================================================

# Unary operations
UNARY_OPS = [
    ("neg", "jnp.negative({x})", "Negate"),
    ("abs", "jnp.abs({x})", "Absolute value"),
    ("exp", "jnp.exp({x})", "Exponential"),
    ("log", "jnp.log(jnp.abs({x}) + 1e-6)", "Log (with safety)"),
    ("sqrt", "jnp.sqrt(jnp.abs({x}))", "Square root"),
    ("sin", "jnp.sin({x})", "Sine"),
    ("cos", "jnp.cos({x})", "Cosine"),
    ("tanh", "jnp.tanh({x})", "Hyperbolic tangent"),
    ("relu", "jax.nn.relu({x})", "ReLU activation"),
    ("sigmoid", "jax.nn.sigmoid({x})", "Sigmoid activation"),
    ("softplus", "jax.nn.softplus({x})", "Softplus"),
    ("square", "{x} ** 2", "Square"),
]

# Binary operations
BINARY_OPS = [
    ("add", "{x} + {y}", "Addition"),
    ("sub", "{x} - {y}", "Subtraction"),
    ("mul", "{x} * {y}", "Multiplication"),
    ("div", "{x} / ({y} + 1e-6)", "Division (safe)"),
    ("pow", "jnp.power(jnp.abs({x}), jnp.abs({y}) % 3 + 1)", "Power"),
    ("max", "jnp.maximum({x}, {y})", "Element-wise max"),
    ("min", "jnp.minimum({x}, {y})", "Element-wise min"),
]

# Reduction operations
REDUCTION_OPS = [
    ("sum", "jnp.sum({x}, axis={axis})", "Sum reduction"),
    ("mean", "jnp.mean({x}, axis={axis})", "Mean reduction"),
    ("max", "jnp.max({x}, axis={axis})", "Max reduction"),
    ("min", "jnp.min({x}, axis={axis})", "Min reduction"),
    ("prod", "jnp.prod({x}, axis={axis})", "Product reduction"),
    ("std", "jnp.std({x}, axis={axis})", "Std reduction"),
]

# Matrix operations
MATRIX_OPS = [
    ("matmul", "jnp.dot({x}, {y})", "Matrix multiplication"),
    ("transpose", "{x}.T", "Transpose"),
    ("trace", "jnp.trace({x})", "Trace"),
]


def random_name(prefix: str = "fn") -> str:
    """Generate a random function name."""
    suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f"{prefix}_{suffix}"


def random_shape(ndim: int = 2, max_size: int = 16) -> Tuple[int, ...]:
    """Generate a random tensor shape."""
    return tuple(random.randint(2, max_size) for _ in range(ndim))


class FunctionGenerator:
    """Generates varied JAX functions for training data synthesis."""
    
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
    
    def generate_unary_fn(self) -> Tuple[str, str, Callable, tuple]:
        """Generate a random unary function."""
        name, template, desc = random.choice(UNARY_OPS)
        fn_name = random_name(name)
        shape = random_shape(random.randint(1, 3))
        
        code = f'''def {fn_name}(x):
    """{desc} operation."""
    return {template.format(x="x")}'''
        
        # Create the actual function
        namespace = {"jnp": jnp, "jax": jax}
        exec(code, namespace)
        fn = namespace[fn_name]
        
        example_args = (jnp.ones(shape),)
        
        return fn_name, code, fn, example_args
    
    def generate_binary_fn(self) -> Tuple[str, str, Callable, tuple]:
        """Generate a random binary function."""
        name, template, desc = random.choice(BINARY_OPS)
        fn_name = random_name(name)
        shape = random_shape(random.randint(1, 3))
        
        code = f'''def {fn_name}(x, y):
    """{desc} operation."""
    return {template.format(x="x", y="y")}'''
        
        namespace = {"jnp": jnp, "jax": jax}
        exec(code, namespace)
        fn = namespace[fn_name]
        
        example_args = (jnp.ones(shape), jnp.ones(shape) * 2)
        
        return fn_name, code, fn, example_args
    
    def generate_reduction_fn(self) -> Tuple[str, str, Callable, tuple]:
        """Generate a random reduction function."""
        name, template, desc = random.choice(REDUCTION_OPS)
        fn_name = random_name(name)
        ndim = random.randint(2, 4)
        shape = random_shape(ndim)
        axis = random.randint(0, ndim - 1)
        
        code = f'''def {fn_name}(x):
    """{desc} along axis {axis}."""
    return {template.format(x="x", axis=axis)}'''
        
        namespace = {"jnp": jnp, "jax": jax}
        exec(code, namespace)
        fn = namespace[fn_name]
        
        example_args = (jnp.ones(shape),)
        
        return fn_name, code, fn, example_args
    
    def generate_matmul_fn(self) -> Tuple[str, str, Callable, tuple]:
        """Generate a matrix multiplication function."""
        fn_name = random_name("matmul")
        m, k, n = random.randint(2, 16), random.randint(2, 16), random.randint(2, 16)
        
        code = f'''def {fn_name}(a, b):
    """Matrix multiplication of shapes ({m},{k}) @ ({k},{n})."""
    return jnp.dot(a, b)'''
        
        namespace = {"jnp": jnp}
        exec(code, namespace)
        fn = namespace[fn_name]
        
        example_args = (jnp.ones((m, k)), jnp.ones((k, n)))
        
        return fn_name, code, fn, example_args
    
    def generate_composite_fn(self) -> Tuple[str, str, Callable, tuple]:
        """Generate a composite function with multiple operations."""
        fn_name = random_name("composite")
        shape = random_shape(2)
        
        # Choose random operations
        ops = random.sample([
            "jax.nn.relu(x)",
            "jnp.tanh(x)",
            "jnp.sin(x)",
            "x ** 2",
            "jnp.exp(-jnp.abs(x))",
        ], k=random.randint(2, 3))
        
        chain = " + ".join(ops)
        
        code = f'''def {fn_name}(x):
    """Composite function with multiple operations."""
    return {chain}'''
        
        namespace = {"jnp": jnp, "jax": jax}
        exec(code, namespace)
        fn = namespace[fn_name]
        
        example_args = (jnp.ones(shape),)
        
        return fn_name, code, fn, example_args
    
    def generate_nn_layer_fn(self) -> Tuple[str, str, Callable, tuple]:
        """Generate a neural network layer function."""
        fn_name = random_name("nn_layer")
        batch = random.randint(1, 8)
        in_feat = random.randint(4, 32)
        out_feat = random.randint(4, 32)
        
        activations = ["jax.nn.relu", "jax.nn.sigmoid", "jax.nn.tanh", "jax.nn.gelu"]
        act = random.choice(activations)
        
        code = f'''def {fn_name}(x, w, b):
    """Neural network layer with {act.split('.')[-1]} activation."""
    return {act}(jnp.dot(x, w) + b)'''
        
        namespace = {"jnp": jnp, "jax": jax}
        exec(code, namespace)
        fn = namespace[fn_name]
        
        example_args = (
            jnp.ones((batch, in_feat)),
            jnp.ones((in_feat, out_feat)),
            jnp.zeros((out_feat,))
        )
        
        return fn_name, code, fn, example_args
    
    def generate_vmap_fn(self) -> Tuple[str, str, Callable, tuple]:
        """Generate a vmapped function."""
        fn_name = random_name("vmap")
        batch = random.randint(2, 8)
        feat = random.randint(4, 16)
        
        inner_ops = random.choice([
            "jnp.dot(a, b)",
            "jnp.sum(a * b)",
            "jnp.maximum(a, b)",
        ])
        
        code = f'''def {fn_name}(a_batch, b_batch):
    """Batched operation using vmap."""
    def inner(a, b):
        return {inner_ops}
    return jax.vmap(inner)(a_batch, b_batch)'''
        
        namespace = {"jnp": jnp, "jax": jax}
        exec(code, namespace)
        fn = namespace[fn_name]
        
        example_args = (jnp.ones((batch, feat)), jnp.ones((batch, feat)))
        
        return fn_name, code, fn, example_args
    
    def generate_random(self) -> Tuple[str, str, Callable, tuple]:
        """Generate a random function from any category."""
        generators = [
            self.generate_unary_fn,
            self.generate_binary_fn,
            self.generate_reduction_fn,
            self.generate_matmul_fn,
            self.generate_composite_fn,
            self.generate_nn_layer_fn,
            self.generate_vmap_fn,
        ]
        
        generator = random.choice(generators)
        return generator()
    
    def generate_batch(self, n: int) -> List[Tuple[str, str, Callable, tuple]]:
        """Generate a batch of random functions."""
        return [self.generate_random() for _ in range(n)]


def demo():
    """Demonstrate the function generator."""
    gen = FunctionGenerator(seed=42)
    
    print("Generating 5 random functions:\n")
    
    for i in range(5):
        fn_name, code, fn, args = gen.generate_random()
        print(f"--- Function {i+1}: {fn_name} ---")
        print(code)
        print(f"Args shapes: {[a.shape for a in args]}")
        
        # Verify it compiles and runs
        result = jax.jit(fn)(*args)
        print(f"Output shape: {result.shape}")
        print()


if __name__ == "__main__":
    demo()
