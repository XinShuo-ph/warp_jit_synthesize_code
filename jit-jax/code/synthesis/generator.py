"""Function generator for creating varied JAX functions programmatically."""
import random
from typing import Tuple, List, Callable
import jax
import jax.numpy as jnp


# ============================================================
# Function Templates
# ============================================================

def make_arithmetic_fn(ops: List[str], name: str) -> Tuple[Callable, str]:
    """Generate arithmetic function from operation sequence."""
    op_map = {
        'add': '+', 'sub': '-', 'mul': '*', 'div': '/',
        'pow2': '**2', 'pow3': '**3', 'neg': '-'
    }
    
    # Build function body
    lines = [f"def {name}(x, y):"]
    expr = "x"
    for i, op in enumerate(ops):
        if op in ['add', 'sub', 'mul', 'div']:
            expr = f"({expr} {op_map[op]} y)"
        elif op in ['pow2', 'pow3']:
            expr = f"({expr}){op_map[op]}"
        elif op == 'neg':
            expr = f"-({expr})"
    lines.append(f"    return {expr}")
    
    source = "\n".join(lines)
    local_ns = {'jnp': jnp}
    exec(source, local_ns)
    fn = local_ns[name]
    return fn, source


def make_unary_fn(ops: List[str], name: str) -> Tuple[Callable, str]:
    """Generate unary function with JAX operations."""
    lines = [f"def {name}(x):"]
    expr = "x"
    for op in ops:
        if op == 'sin':
            expr = f"jnp.sin({expr})"
        elif op == 'cos':
            expr = f"jnp.cos({expr})"
        elif op == 'exp':
            expr = f"jnp.exp({expr})"
        elif op == 'log':
            expr = f"jnp.log(jnp.abs({expr}) + 1e-6)"
        elif op == 'tanh':
            expr = f"jnp.tanh({expr})"
        elif op == 'relu':
            expr = f"jnp.maximum(0, {expr})"
        elif op == 'sigmoid':
            expr = f"(1 / (1 + jnp.exp(-{expr})))"
        elif op == 'sqrt':
            expr = f"jnp.sqrt(jnp.abs({expr}))"
        elif op == 'square':
            expr = f"({expr})**2"
    lines.append(f"    return {expr}")
    
    source = "\n".join(lines)
    local_ns = {'jnp': jnp}
    exec(source, local_ns)
    fn = local_ns[name]
    return fn, source


def make_reduction_fn(reduce_op: str, pre_ops: List[str], name: str) -> Tuple[Callable, str]:
    """Generate reduction function."""
    lines = [f"def {name}(x):"]
    
    # Pre-processing
    expr = "x"
    for op in pre_ops:
        if op == 'square':
            expr = f"({expr})**2"
        elif op == 'abs':
            expr = f"jnp.abs({expr})"
        elif op == 'sin':
            expr = f"jnp.sin({expr})"
    
    # Reduction
    if reduce_op == 'sum':
        expr = f"jnp.sum({expr})"
    elif reduce_op == 'mean':
        expr = f"jnp.mean({expr})"
    elif reduce_op == 'max':
        expr = f"jnp.max({expr})"
    elif reduce_op == 'min':
        expr = f"jnp.min({expr})"
    elif reduce_op == 'prod':
        expr = f"jnp.prod({expr})"
    
    lines.append(f"    return {expr}")
    
    source = "\n".join(lines)
    local_ns = {'jnp': jnp}
    exec(source, local_ns)
    fn = local_ns[name]
    return fn, source


def make_matrix_fn(ops: List[str], name: str) -> Tuple[Callable, str]:
    """Generate matrix operation function."""
    lines = [f"def {name}(A, x):"]
    
    expr = "A @ x"
    for op in ops:
        if op == 'relu':
            expr = f"jnp.maximum(0, {expr})"
        elif op == 'tanh':
            expr = f"jnp.tanh({expr})"
        elif op == 'softmax':
            expr = f"jax.nn.softmax({expr})"
        elif op == 'normalize':
            expr = f"({expr}) / (jnp.linalg.norm({expr}) + 1e-6)"
        elif op == 'square':
            expr = f"({expr})**2"
    
    lines.append(f"    return {expr}")
    
    source = "\n".join(lines)
    local_ns = {'jnp': jnp, 'jax': jax}
    exec(source, local_ns)
    fn = local_ns[name]
    return fn, source


def make_compound_fn(name: str, seed: int) -> Tuple[Callable, str]:
    """Generate compound function with multiple operations."""
    random.seed(seed)
    
    ops = random.sample(['sin', 'cos', 'exp', 'square', 'tanh'], k=random.randint(2, 4))
    combine = random.choice(['add', 'mul'])
    
    lines = [f"def {name}(x, y):"]
    
    # Build two branches
    expr1 = "x"
    expr2 = "y"
    for i, op in enumerate(ops):
        if i % 2 == 0:
            if op == 'sin':
                expr1 = f"jnp.sin({expr1})"
            elif op == 'cos':
                expr1 = f"jnp.cos({expr1})"
            elif op == 'exp':
                expr1 = f"jnp.exp(-jnp.abs({expr1}))"
            elif op == 'square':
                expr1 = f"({expr1})**2"
            elif op == 'tanh':
                expr1 = f"jnp.tanh({expr1})"
        else:
            if op == 'sin':
                expr2 = f"jnp.sin({expr2})"
            elif op == 'cos':
                expr2 = f"jnp.cos({expr2})"
            elif op == 'exp':
                expr2 = f"jnp.exp(-jnp.abs({expr2}))"
            elif op == 'square':
                expr2 = f"({expr2})**2"
            elif op == 'tanh':
                expr2 = f"jnp.tanh({expr2})"
    
    if combine == 'add':
        final = f"({expr1}) + ({expr2})"
    else:
        final = f"({expr1}) * ({expr2})"
    
    lines.append(f"    return {final}")
    
    source = "\n".join(lines)
    local_ns = {'jnp': jnp}
    exec(source, local_ns)
    fn = local_ns[name]
    return fn, source


# ============================================================
# Generator Class
# ============================================================

class FunctionGenerator:
    """Generates varied JAX functions for training data."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.counter = 0
        random.seed(seed)
    
    def _next_name(self, prefix: str = "fn") -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"
    
    def generate_arithmetic(self) -> Tuple[Callable, str, dict]:
        """Generate arithmetic function."""
        ops = random.sample(['add', 'sub', 'mul', 'pow2'], k=random.randint(2, 4))
        name = self._next_name("arith")
        fn, source = make_arithmetic_fn(ops, name)
        
        # Generate test inputs
        key = jax.random.PRNGKey(self.counter)
        x = jax.random.normal(key, (4,))
        y = jax.random.normal(key, (4,)) + 0.1  # Avoid division by zero
        
        return fn, source, {'args': (x, y), 'type': 'arithmetic'}
    
    def generate_unary(self) -> Tuple[Callable, str, dict]:
        """Generate unary function."""
        ops = random.sample(['sin', 'cos', 'exp', 'tanh', 'relu', 'square'], k=random.randint(1, 3))
        name = self._next_name("unary")
        fn, source = make_unary_fn(ops, name)
        
        key = jax.random.PRNGKey(self.counter)
        x = jax.random.normal(key, (8,))
        
        return fn, source, {'args': (x,), 'type': 'unary'}
    
    def generate_reduction(self) -> Tuple[Callable, str, dict]:
        """Generate reduction function."""
        reduce_op = random.choice(['sum', 'mean', 'max', 'min'])
        pre_ops = random.sample(['square', 'abs', 'sin'], k=random.randint(0, 2))
        name = self._next_name("reduce")
        fn, source = make_reduction_fn(reduce_op, pre_ops, name)
        
        key = jax.random.PRNGKey(self.counter)
        x = jax.random.normal(key, (16,))
        
        return fn, source, {'args': (x,), 'type': 'reduction'}
    
    def generate_matrix(self) -> Tuple[Callable, str, dict]:
        """Generate matrix operation function."""
        ops = random.sample(['relu', 'tanh', 'normalize', 'square'], k=random.randint(1, 2))
        name = self._next_name("matrix")
        fn, source = make_matrix_fn(ops, name)
        
        key = jax.random.PRNGKey(self.counter)
        A = jax.random.normal(key, (4, 8))
        x = jax.random.normal(key, (8,))
        
        return fn, source, {'args': (A, x), 'type': 'matrix'}
    
    def generate_compound(self) -> Tuple[Callable, str, dict]:
        """Generate compound function."""
        name = self._next_name("compound")
        fn, source = make_compound_fn(name, self.counter)
        
        key = jax.random.PRNGKey(self.counter)
        x = jax.random.normal(key, (6,))
        y = jax.random.normal(key, (6,))
        
        return fn, source, {'args': (x, y), 'type': 'compound'}
    
    def generate_random(self) -> Tuple[Callable, str, dict]:
        """Generate a random function type."""
        generators = [
            self.generate_arithmetic,
            self.generate_unary,
            self.generate_reduction,
            self.generate_matrix,
            self.generate_compound,
        ]
        return random.choice(generators)()
    
    def generate_batch(self, n: int) -> List[Tuple[Callable, str, dict]]:
        """Generate n random functions."""
        return [self.generate_random() for _ in range(n)]


if __name__ == "__main__":
    gen = FunctionGenerator(seed=42)
    
    print("Generating 20 sample functions:\n")
    
    for i in range(20):
        fn, source, meta = gen.generate_random()
        print(f"--- Function {i+1} ({meta['type']}) ---")
        print(source)
        # Test it works
        result = fn(*meta['args'])
        print(f"Output shape: {result.shape if hasattr(result, 'shape') else 'scalar'}\n")
    
    print("✓ Generated 20 unique functions successfully!")
