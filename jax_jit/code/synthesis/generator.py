"""Programmatic function generator for varied Python→StableHLO pairs."""
import random
import string
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import textwrap

import jax.numpy as jnp


@dataclass
class FunctionSpec:
    """Specification for a generated JAX function."""
    name: str
    params: List[tuple]  # [(name, description), ...]
    body_lines: List[str]
    imports: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


# Type specifications for JAX
SCALAR_DTYPES = ["float", "int"]
ARRAY_SHAPES = [(4,), (8,), (16,), (32,)]
MATRIX_SHAPES = [(2, 2), (3, 3), (4, 4), (2, 4), (4, 2)]
TENSOR_SHAPES = [(2, 2, 2), (2, 3, 4)]

# Operations by category
BINARY_OPS = ["+", "-", "*", "/"]
COMPARE_OPS = [">", "<", ">=", "<=", "==", "!="]
UNARY_FUNCS = ["jnp.sin", "jnp.cos", "jnp.sqrt", "jnp.abs", "jnp.exp", "jnp.log", "jnp.tanh"]
BINARY_FUNCS = ["jnp.add", "jnp.subtract", "jnp.multiply", "jnp.divide", "jnp.power", "jnp.maximum", "jnp.minimum"]
REDUCTION_FUNCS = ["jnp.sum", "jnp.mean", "jnp.max", "jnp.min", "jnp.prod"]
LINEAR_ALGEBRA = ["jnp.dot", "jnp.matmul", "jnp.outer"]


def random_name(prefix: str = "func") -> str:
    """Generate a random function name."""
    suffix = ''.join(random.choices(string.ascii_lowercase, k=6))
    return f"{prefix}_{suffix}"


def random_float(low: float = -10.0, high: float = 10.0) -> str:
    """Generate a random float literal."""
    val = round(random.uniform(low, high), 2)
    return f"{val}"


def random_int(low: int = 1, high: int = 100) -> str:
    """Generate a random int literal."""
    return str(random.randint(low, high))


class FunctionGenerator:
    """Generator for varied JAX functions."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.category_generators = {
            'arithmetic': self.gen_arithmetic,
            'conditional': self.gen_conditional,
            'reduction': self.gen_reduction,
            'matrix': self.gen_matrix_ops,
            'elementwise': self.gen_elementwise,
            'broadcasting': self.gen_broadcasting,
            'composite': self.gen_composite,
        }
    
    def gen_arithmetic(self) -> FunctionSpec:
        """Generate a simple arithmetic function."""
        name = random_name("arith")
        op1 = random.choice(BINARY_OPS)
        op2 = random.choice(BINARY_OPS)
        const = random_float()
        
        # Use same size to ensure compatibility
        params = [
            ("x", "array_same"),
            ("y", "array_same"),
        ]
        
        body = [
            f"return (x {op1} y) {op2} {const}",
        ]
        
        return FunctionSpec(
            name=name,
            params=params,
            body_lines=body,
            docstring=f"Compute (x {op1} y) {op2} {const}"
        )
    
    def gen_conditional(self) -> FunctionSpec:
        """Generate a function with conditional logic."""
        name = random_name("cond")
        cmp_op = random.choice(COMPARE_OPS)
        threshold = random_float()
        scale_true = random_float()
        scale_false = random_float()
        
        params = [("x", "array")]
        
        body = [
            f"return jnp.where(x {cmp_op} {threshold}, x * {scale_true}, x * {scale_false})",
        ]
        
        return FunctionSpec(
            name=name,
            params=params,
            body_lines=body,
            docstring=f"Scale x based on threshold {threshold}"
        )
    
    def gen_reduction(self) -> FunctionSpec:
        """Generate a reduction function."""
        name = random_name("reduce")
        reduction = random.choice(REDUCTION_FUNCS)
        pre_op = random.choice(["x * x", "x + 1.0", "x - 1.0", "x * 2.0", "jnp.abs(x)", "x"])
        post_scale = random_float(0.1, 5.0)
        
        params = [("x", "array")]
        
        body = [
            f"return {reduction}({pre_op}) * {post_scale}",
        ]
        
        return FunctionSpec(
            name=name,
            params=params,
            body_lines=body,
            docstring=f"Apply {reduction} to {pre_op}"
        )
    
    def gen_matrix_ops(self) -> FunctionSpec:
        """Generate matrix operations."""
        name = random_name("matrix")
        op = random.choice(LINEAR_ALGEBRA)
        
        if op == "jnp.dot" or op == "jnp.matmul":
            # Use compatible matrix sizes
            params = [("A", "matrix_compatible"), ("B", "matrix_compatible")]
            body = [f"return {op}(A, B)"]
            docstring = "Matrix multiplication"
        elif op == "jnp.outer":
            params = [("x", "vector"), ("y", "vector")]
            body = [f"return {op}(x, y)"]
            docstring = "Outer product"
        else:
            params = [("A", "matrix")]
            body = [f"return {op}(A)"]
            docstring = f"Apply {op}"
        
        return FunctionSpec(
            name=name,
            params=params,
            body_lines=body,
            docstring=docstring
        )
    
    def gen_elementwise(self) -> FunctionSpec:
        """Generate element-wise operations."""
        name = random_name("elem")
        func = random.choice(UNARY_FUNCS)
        pre_scale = random_float(0.1, 2.0)
        post_scale = random_float(0.1, 2.0)
        
        params = [("x", "array")]
        
        body = [
            f"return {func}(x * {pre_scale}) * {post_scale}",
        ]
        
        return FunctionSpec(
            name=name,
            params=params,
            body_lines=body,
            docstring=f"Element-wise {func}"
        )
    
    def gen_broadcasting(self) -> FunctionSpec:
        """Generate broadcasting operations."""
        name = random_name("bcast")
        op = random.choice(BINARY_OPS)
        const = random_float()
        
        params = [
            ("scalar", "float"),
            ("vector", "array"),
        ]
        
        body = [
            f"return scalar {op} vector {op} {const}",
        ]
        
        return FunctionSpec(
            name=name,
            params=params,
            body_lines=body,
            docstring="Broadcasting operations"
        )
    
    def gen_composite(self) -> FunctionSpec:
        """Generate composite operations."""
        name = random_name("composite")
        
        # Create a more complex computation
        op1 = random.choice(BINARY_OPS)
        op2 = random.choice(BINARY_OPS)
        func1 = random.choice(UNARY_FUNCS)
        func2 = random.choice(UNARY_FUNCS)
        const1 = random_float()
        const2 = random_float()
        
        # Use same shape for compatibility
        params = [("x", "array_same"), ("y", "array_same")]
        
        body = [
            f"temp1 = {func1}(x) {op1} {const1}",
            f"temp2 = {func2}(y) {op2} {const2}",
            f"return temp1 * temp2",
        ]
        
        return FunctionSpec(
            name=name,
            params=params,
            body_lines=body,
            docstring="Composite operation"
        )
    
    def generate(self, category: Optional[str] = None) -> FunctionSpec:
        """
        Generate a function specification.
        
        Args:
            category: Optional category to generate from. If None, random.
        
        Returns:
            FunctionSpec
        """
        if category is None:
            category = random.choice(list(self.category_generators.keys()))
        
        if category not in self.category_generators:
            raise ValueError(f"Unknown category: {category}")
        
        return self.category_generators[category]()
    
    def generate_batch(self, count: int, categories: Optional[List[str]] = None) -> List[FunctionSpec]:
        """
        Generate a batch of function specifications.
        
        Args:
            count: Number of functions to generate
            categories: Optional list of categories to sample from
        
        Returns:
            List of FunctionSpec
        """
        if categories is None:
            categories = list(self.category_generators.keys())
        
        specs = []
        for _ in range(count):
            category = random.choice(categories)
            specs.append(self.generate(category))
        
        return specs


def spec_to_code(spec: FunctionSpec) -> str:
    """
    Convert a FunctionSpec to executable Python code.
    
    Args:
        spec: The FunctionSpec to convert
    
    Returns:
        String containing the complete function code
    """
    lines = []
    
    # Add imports if any
    if spec.imports:
        for imp in spec.imports:
            lines.append(imp)
        lines.append("")
    
    # Function signature
    param_str = ", ".join([name for name, _ in spec.params])
    lines.append(f"def {spec.name}({param_str}):")
    
    # Docstring
    if spec.docstring:
        lines.append(f'    """{spec.docstring}"""')
    
    # Body
    for body_line in spec.body_lines:
        lines.append(f"    {body_line}")
    
    return "\n".join(lines)


def spec_to_callable(spec: FunctionSpec) -> Callable:
    """
    Convert a FunctionSpec to an executable callable function.
    
    Args:
        spec: The FunctionSpec to convert
    
    Returns:
        Callable function
    """
    code = spec_to_code(spec)
    
    # Create a namespace with jax.numpy
    namespace = {'jnp': jnp}
    
    # Execute the code to define the function
    exec(code, namespace)
    
    # Return the function
    return namespace[spec.name]


def generate_example_inputs(params: List[tuple]) -> List:
    """
    Generate example inputs for a function based on its parameters.
    
    Args:
        params: List of (name, description) tuples
    
    Returns:
        List of example input arrays
    """
    inputs = []
    
    # Track shape for "same" descriptors
    same_shape = None
    compatible_matrix_size = None
    
    for name, desc in params:
        if "scalar" in desc.lower() or "float" in desc.lower():
            inputs.append(random.uniform(0.1, 5.0))
        elif "vector" in desc.lower():
            size = random.choice([4, 8, 16])
            inputs.append(jnp.array([random.uniform(-5, 5) for _ in range(size)]))
        elif "array_same" in desc.lower():
            # Use same shape for all "array_same" parameters
            if same_shape is None:
                same_shape = random.choice(ARRAY_SHAPES)
            inputs.append(jnp.array([random.uniform(-5, 5) for _ in range(same_shape[0])]))
        elif "matrix_compatible" in desc.lower():
            # Generate compatible matrices for multiplication
            if compatible_matrix_size is None:
                compatible_matrix_size = random.choice([2, 3, 4])
            # First matrix: (n, m), second matrix: (m, k) -> result (n, k)
            if not inputs:  # First matrix
                shape = (compatible_matrix_size, compatible_matrix_size)
                inputs.append(jnp.array([[random.uniform(-5, 5) for _ in range(shape[1])]
                                         for _ in range(shape[0])]))
            else:  # Second matrix - must match first matrix's columns
                prev_shape = inputs[-1].shape
                shape = (prev_shape[1], compatible_matrix_size)
                inputs.append(jnp.array([[random.uniform(-5, 5) for _ in range(shape[1])]
                                         for _ in range(shape[0])]))
        elif "matrix" in desc.lower():
            shape = random.choice(MATRIX_SHAPES)
            inputs.append(jnp.array([[random.uniform(-5, 5) for _ in range(shape[1])]
                                     for _ in range(shape[0])]))
        else:  # default to array
            shape = random.choice(ARRAY_SHAPES)
            inputs.append(jnp.array([random.uniform(-5, 5) for _ in range(shape[0])]))
    
    return inputs
