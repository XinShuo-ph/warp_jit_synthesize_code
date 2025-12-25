"""Automated Kernel Generator for Synthesis Pipeline.

Generates diverse warp kernels programmatically for IR extraction.

Design:
- Multiple kernel patterns (arithmetic, array ops, conditionals, loops, vectors)
- Parameterized generation (different operators, constants, types)
- Ensures diversity through randomization
"""

import warp as wp
import random
import itertools
from typing import Callable, List, Tuple


class KernelGenerator:
    """Generates diverse warp kernel functions programmatically."""
    
    def __init__(self, seed=42):
        """Initialize generator with random seed for reproducibility."""
        random.seed(seed)
        self.kernel_count = 0
    
    def _make_unique_name(self, pattern: str) -> str:
        """Generate unique kernel name."""
        self.kernel_count += 1
        return f"{pattern}_gen_{self.kernel_count}"
    
    def generate_arithmetic_kernel(self, ops: List[str] = None) -> Tuple[Callable, str]:
        """Generate kernel with arithmetic operations.
        
        Args:
            ops: List of operations to include (add, sub, mul, div)
            
        Returns:
            Tuple of (kernel_function, description)
        """
        if ops is None:
            ops_choice = random.randint(0, 3)
        else:
            ops_choice = 0  # Use provided ops
        
        # Pre-defined arithmetic patterns
        if ops_choice == 0:
            @wp.kernel
            def arithmetic_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
                tid = wp.tid()
                result[tid] = a[tid] * 2.0 + 1.0
            desc = "Arithmetic: mul, add"
        elif ops_choice == 1:
            @wp.kernel
            def arithmetic_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
                tid = wp.tid()
                result[tid] = a[tid] * 3.0 - 0.5
            desc = "Arithmetic: mul, sub"
        elif ops_choice == 2:
            @wp.kernel
            def arithmetic_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
                tid = wp.tid()
                result[tid] = (a[tid] + 1.0) / 2.0
            desc = "Arithmetic: add, div"
        else:
            @wp.kernel
            def arithmetic_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
                tid = wp.tid()
                result[tid] = a[tid] * 1.5 + 2.0 - 0.3
            desc = "Arithmetic: mul, add, sub"
        
        return arithmetic_kernel, desc
    
    def generate_array_indexing_kernel(self) -> Tuple[Callable, str]:
        """Generate kernel with array indexing patterns."""
        
        offset = random.randint(-2, 2)
        scale = random.choice([1, 2, 3])
        
        @wp.kernel
        def array_indexing_kernel(a: wp.array(dtype=float), indices: wp.array(dtype=int), 
                                   result: wp.array(dtype=float)):
            tid = wp.tid()
            idx = indices[tid]
            result[tid] = a[idx] * float(scale)
        
        desc = f"Array indexing with scale={scale}"
        return array_indexing_kernel, desc
    
    def generate_conditional_kernel(self, threshold: float = None) -> Tuple[Callable, str]:
        """Generate kernel with conditional logic."""
        
        if threshold is None:
            threshold = random.uniform(-1.0, 1.0)
        
        scale_true = random.uniform(1.5, 3.0)
        scale_false = random.uniform(0.1, 0.9)
        
        # Need to use closures to capture values
        def make_kernel(thresh, s_true, s_false):
            @wp.kernel
            def conditional_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
                tid = wp.tid()
                val = a[tid]
                if val > thresh:
                    result[tid] = val * s_true
                else:
                    result[tid] = val * s_false
            return conditional_kernel
        
        kernel = make_kernel(threshold, scale_true, scale_false)
        desc = f"Conditional: threshold={threshold:.2f}"
        return kernel, desc
    
    def generate_loop_kernel(self, iterations: int = None) -> Tuple[Callable, str]:
        """Generate kernel with loop."""
        
        if iterations is None:
            iterations = random.randint(2, 5)
        
        def make_kernel(iters):
            @wp.kernel
            def loop_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
                tid = wp.tid()
                sum_val = 0.0
                for i in range(iters):
                    sum_val = sum_val + a[tid]
                result[tid] = sum_val
            return loop_kernel
        
        kernel = make_kernel(iterations)
        desc = f"Loop: {iterations} iterations"
        return kernel, desc
    
    def generate_vector_kernel(self) -> Tuple[Callable, str]:
        """Generate kernel with vector operations."""
        
        op = random.choice(['length', 'dot', 'normalize'])
        
        if op == 'length':
            @wp.kernel
            def vector_kernel(a: wp.array(dtype=wp.vec3), result: wp.array(dtype=float)):
                tid = wp.tid()
                v = a[tid]
                result[tid] = wp.length(v)
            desc = "Vector: length"
        elif op == 'dot':
            @wp.kernel
            def vector_kernel(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), 
                             result: wp.array(dtype=float)):
                tid = wp.tid()
                v1 = a[tid]
                v2 = b[tid]
                result[tid] = wp.dot(v1, v2)
            desc = "Vector: dot product"
        else:  # normalize
            @wp.kernel
            def vector_kernel(a: wp.array(dtype=wp.vec3), result: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                v = a[tid]
                result[tid] = wp.normalize(v)
            desc = "Vector: normalize"
        
        return vector_kernel, desc
    
    def generate_math_kernel(self) -> Tuple[Callable, str]:
        """Generate kernel with math functions."""
        
        func = random.choice(['sin', 'cos', 'exp', 'sqrt', 'abs'])
        scale = random.uniform(0.5, 2.0)
        
        def make_kernel(f, s):
            if f == 'sin':
                @wp.kernel
                def math_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
                    tid = wp.tid()
                    result[tid] = wp.sin(a[tid] * s)
            elif f == 'cos':
                @wp.kernel
                def math_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
                    tid = wp.tid()
                    result[tid] = wp.cos(a[tid] * s)
            elif f == 'exp':
                @wp.kernel
                def math_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
                    tid = wp.tid()
                    result[tid] = wp.exp(a[tid] * s)
            elif f == 'sqrt':
                @wp.kernel
                def math_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
                    tid = wp.tid()
                    result[tid] = wp.sqrt(wp.abs(a[tid]) * s)
            else:  # abs
                @wp.kernel
                def math_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
                    tid = wp.tid()
                    result[tid] = wp.abs(a[tid] * s)
            return math_kernel
        
        kernel = make_kernel(func, scale)
        desc = f"Math: {func} with scale={scale:.2f}"
        return kernel, desc
    
    def generate_multi_op_kernel(self) -> Tuple[Callable, str]:
        """Generate kernel with multiple complex operations."""
        
        @wp.kernel
        def multi_op_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), 
                           c: wp.array(dtype=float)):
            tid = wp.tid()
            x = a[tid]
            y = b[tid]
            temp = x * x + y * y
            c[tid] = wp.sqrt(temp)
        
        desc = "Multi-op: sqrt(x^2 + y^2)"
        return multi_op_kernel, desc
    
    def generate_batch(self, count: int = 10) -> List[Tuple[Callable, str]]:
        """Generate a batch of diverse kernels.
        
        Args:
            count: Number of kernels to generate
            
        Returns:
            List of (kernel, description) tuples
        """
        generators = [
            self.generate_arithmetic_kernel,
            self.generate_array_indexing_kernel,
            self.generate_conditional_kernel,
            self.generate_loop_kernel,
            self.generate_vector_kernel,
            self.generate_math_kernel,
            self.generate_multi_op_kernel,
        ]
        
        kernels = []
        for i in range(count):
            # Choose generator with some variety
            gen = random.choice(generators)
            try:
                kernel, desc = gen()
                kernels.append((kernel, desc))
            except Exception as e:
                print(f"Warning: Failed to generate kernel: {e}")
        
        return kernels


# Test the generator
if __name__ == "__main__":
    wp.init()
    
    gen = KernelGenerator(seed=42)
    
    print("Testing Kernel Generator")
    print("=" * 70)
    
    # Test each generator
    generators = [
        ("Arithmetic", gen.generate_arithmetic_kernel),
        ("Array Indexing", gen.generate_array_indexing_kernel),
        ("Conditional", gen.generate_conditional_kernel),
        ("Loop", gen.generate_loop_kernel),
        ("Vector", gen.generate_vector_kernel),
        ("Math", gen.generate_math_kernel),
        ("Multi-op", gen.generate_multi_op_kernel),
    ]
    
    for name, generator in generators:
        try:
            kernel, desc = generator()
            print(f"✓ {name}: {desc}")
        except Exception as e:
            print(f"✗ {name}: FAILED - {e}")
    
    print("\n" + "=" * 70)
    print("Generating batch of 10 kernels...")
    batch = gen.generate_batch(10)
    print(f"Generated {len(batch)} kernels")
    
    for i, (kernel, desc) in enumerate(batch, 1):
        print(f"  {i}. {desc}")
