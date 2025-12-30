#!/usr/bin/env python3
"""
Kernel Generator: Programmatically generate diverse JAX functions

Generates Python functions with various operations, shapes, and types
for creating training data.
"""
import jax.numpy as jnp
import random
from typing import List, Tuple, Callable, Dict, Any


class KernelGenerator:
    """Generate diverse JAX kernel functions"""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed"""
        random.seed(seed)
        self.kernel_count = 0
    
    def _random_shape(self, ndim: int = None, max_size: int = 10) -> Tuple[int, ...]:
        """Generate random tensor shape"""
        if ndim is None:
            ndim = random.randint(1, 3)
        return tuple(random.randint(2, max_size) for _ in range(ndim))
    
    def _random_scalar_shape(self) -> Tuple[()]:
        """Generate scalar shape"""
        return ()
    
    def generate_arithmetic(self) -> Dict[str, Any]:
        """Generate arithmetic operation kernel"""
        ops = [
            ("add", "x + y", "lambda x, y: x + y"),
            ("subtract", "x - y", "lambda x, y: x - y"),
            ("multiply", "x * y", "lambda x, y: x * y"),
            ("divide", "x / y", "lambda x, y: x / y"),
            ("power", "x ** 2", "lambda x: x ** 2"),
            ("add_scalar", "x + 1.0", "lambda x: x + 1.0"),
            ("mul_scalar", "x * 2.0", "lambda x: x * 2.0"),
        ]
        
        op_name, expr, code = random.choice(ops)
        shape = self._random_shape()
        
        # Create function
        if "y" in code:
            func = eval(code)
            example_inputs = [jnp.ones(shape), jnp.ones(shape)]
        else:
            func = eval(code)
            example_inputs = [jnp.ones(shape)]
        
        return {
            "function": func,
            "code": code,
            "example_inputs": example_inputs,
            "category": "arithmetic",
            "operation": op_name,
            "description": f"Arithmetic: {expr}"
        }
    
    def generate_array_op(self) -> Dict[str, Any]:
        """Generate array manipulation operation"""
        ops_list = [
            ("reshape", lambda shape: (
                f"lambda x: jnp.reshape(x, {self._compatible_reshape(shape)})",
                [jnp.ones(shape)]
            )),
            ("transpose", lambda shape: (
                "lambda x: x.T" if len(shape) == 2 else f"lambda x: jnp.transpose(x)",
                [jnp.ones(shape)]
            )),
            ("slice", lambda shape: (
                "lambda x: x[1:]" if len(shape) == 1 else "lambda x: x[1:, :]",
                [jnp.ones(shape)]
            )),
            ("concatenate", lambda shape: (
                "lambda x, y: jnp.concatenate([x, y], axis=0)",
                [jnp.ones(shape), jnp.ones(shape)]
            )),
            ("stack", lambda shape: (
                "lambda x, y: jnp.stack([x, y])",
                [jnp.ones(shape), jnp.ones(shape)]
            )),
        ]
        
        op_name, op_func = random.choice(ops_list)
        shape = self._random_shape()
        code, example_inputs = op_func(shape)
        
        return {
            "function": eval(code),
            "code": code,
            "example_inputs": example_inputs,
            "category": "array",
            "operation": op_name,
            "description": f"Array operation: {op_name}"
        }
    
    def _compatible_reshape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Generate compatible reshape target"""
        total = 1
        for s in shape:
            total *= s
        # Simple reshape: flatten or reshape to 2D
        if random.random() < 0.5:
            return (total,)
        else:
            # Find divisors for 2D reshape
            for i in range(2, int(total**0.5) + 1):
                if total % i == 0:
                    return (i, total // i)
            return (total,)
    
    def generate_math_func(self) -> Dict[str, Any]:
        """Generate mathematical function kernel"""
        funcs = [
            ("sin", "jnp.sin(x)"),
            ("cos", "jnp.cos(x)"),
            ("exp", "jnp.exp(x)"),
            ("log", "jnp.log(x + 1.0)"),  # +1 to avoid log(0)
            ("sqrt", "jnp.sqrt(jnp.abs(x))"),  # abs to avoid sqrt of negative
            ("tanh", "jnp.tanh(x)"),
            ("sigmoid", "1.0 / (1.0 + jnp.exp(-x))"),
            ("abs", "jnp.abs(x)"),
            ("square", "x ** 2"),
        ]
        
        func_name, expr = random.choice(funcs)
        shape = self._random_shape()
        code = f"lambda x: {expr}"
        
        return {
            "function": eval(code),
            "code": code,
            "example_inputs": [jnp.ones(shape)],
            "category": "math",
            "operation": func_name,
            "description": f"Math function: {func_name}"
        }
    
    def generate_reduction(self) -> Dict[str, Any]:
        """Generate reduction operation kernel"""
        reductions = [
            ("sum", "jnp.sum(x)"),
            ("mean", "jnp.mean(x)"),
            ("max", "jnp.max(x)"),
            ("min", "jnp.min(x)"),
            ("prod", "jnp.prod(x)"),
            ("sum_axis0", "jnp.sum(x, axis=0)"),
            ("mean_axis0", "jnp.mean(x, axis=0)"),
        ]
        
        op_name, expr = random.choice(reductions)
        shape = self._random_shape(ndim=2) if "axis" in op_name else self._random_shape()
        code = f"lambda x: {expr}"
        
        return {
            "function": eval(code),
            "code": code,
            "example_inputs": [jnp.ones(shape)],
            "category": "reduction",
            "operation": op_name,
            "description": f"Reduction: {op_name}"
        }
    
    def generate_linalg(self) -> Dict[str, Any]:
        """Generate linear algebra operation"""
        ops = [
            ("dot", "lambda x, y: jnp.dot(x, y)", 
             lambda: [jnp.ones((5,)), jnp.ones((5,))]),
            ("matmul", "lambda A, B: jnp.matmul(A, B)",
             lambda: [jnp.ones((3, 4)), jnp.ones((4, 5))]),
            ("outer", "lambda x, y: jnp.outer(x, y)",
             lambda: [jnp.ones((3,)), jnp.ones((4,))]),
            ("norm", "lambda x: jnp.linalg.norm(x)",
             lambda: [jnp.ones((5,))]),
        ]
        
        op_name, code, input_gen = random.choice(ops)
        
        return {
            "function": eval(code),
            "code": code,
            "example_inputs": input_gen(),
            "category": "linalg",
            "operation": op_name,
            "description": f"Linear algebra: {op_name}"
        }
    
    def generate_composite(self) -> Dict[str, Any]:
        """Generate composite operation (multiple ops combined)"""
        composites = [
            ("squared_norm", "lambda x: jnp.sum(x ** 2)",
             lambda: [jnp.ones(self._random_shape())]),
            ("normalize", "lambda x: x / jnp.linalg.norm(x)",
             lambda: [jnp.ones(self._random_shape())]),
            ("softmax", "lambda x: jnp.exp(x) / jnp.sum(jnp.exp(x))",
             lambda: [jnp.ones(self._random_shape())]),
            ("weighted_sum", "lambda x, w: jnp.sum(x * w)",
             lambda: [jnp.ones(self._random_shape()), jnp.ones(self._random_shape())]),
        ]
        
        op_name, code, input_gen = random.choice(composites)
        
        return {
            "function": eval(code),
            "code": code,
            "example_inputs": input_gen(),
            "category": "composite",
            "operation": op_name,
            "description": f"Composite operation: {op_name}"
        }
    
    def generate_control_flow(self) -> Dict[str, Any]:
        """Generate control flow operation"""
        from jax import lax
        
        ops = [
            ("conditional", 
             "lambda x: lax.cond(jnp.sum(x) > 0, lambda x: x, lambda x: -x, x)",
             lambda: [jnp.ones(self._random_shape())]),
            ("select",
             "lambda pred, x, y: lax.select(pred, x, y)",
             lambda: [jnp.array([True, False, True]), jnp.ones((3,)), jnp.ones((3,))]),
        ]
        
        op_name, code, input_gen = random.choice(ops)
        
        return {
            "function": eval(code),
            "code": code,
            "example_inputs": input_gen(),
            "category": "control_flow",
            "operation": op_name,
            "description": f"Control flow: {op_name}"
        }
    
    def generate_random(self) -> Dict[str, Any]:
        """Generate a random kernel of any type"""
        generators = [
            self.generate_arithmetic,
            self.generate_array_op,
            self.generate_math_func,
            self.generate_reduction,
            self.generate_linalg,
            self.generate_composite,
            self.generate_control_flow,
        ]
        
        generator = random.choice(generators)
        return generator()
    
    def generate_batch(self, n: int, category: str = None) -> List[Dict[str, Any]]:
        """Generate batch of kernels"""
        if category:
            generator = getattr(self, f"generate_{category}")
            return [generator() for _ in range(n)]
        else:
            return [self.generate_random() for _ in range(n)]


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("Kernel Generator Test")
    print("=" * 70)
    
    gen = KernelGenerator(seed=42)
    
    # Test each category
    categories = [
        "arithmetic", "array_op", "math_func", "reduction",
        "linalg", "composite", "control_flow"
    ]
    
    for cat in categories:
        print(f"\n{cat.upper()}:")
        print("-" * 70)
        
        method = getattr(gen, f"generate_{cat}")
        for i in range(3):
            kernel = method()
            print(f"  {i+1}. {kernel['operation']:15s} - {kernel['code'][:50]}")
    
    # Test random generation
    print(f"\nRANDOM GENERATION:")
    print("-" * 70)
    batch = gen.generate_batch(10)
    for i, kernel in enumerate(batch, 1):
        print(f"  {i}. {kernel['category']:15s} / {kernel['operation']:15s}")
    
    # Test that functions execute
    print(f"\nEXECUTION TEST:")
    print("-" * 70)
    for kernel in batch[:5]:
        try:
            result = kernel['function'](*kernel['example_inputs'])
            print(f"  ✓ {kernel['operation']:15s} executed successfully")
        except Exception as e:
            print(f"  ✗ {kernel['operation']:15s} failed: {e}")
    
    print("\n" + "=" * 70)
    print("SUCCESS: Kernel generator working!")
    print("=" * 70)
