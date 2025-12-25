"""
Kernel Generator for Synthesis Pipeline

Programmatically generates varied Warp kernels for training data generation.
Supports multiple kernel types and complexity levels.

Note: Since Warp requires kernels to be defined in files (not via exec()),
we generate source code strings that can be written to files and imported.
"""

import warp as wp
import random
import os
import tempfile
import importlib.util
import sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class KernelSpec:
    """Specification for a generated kernel."""
    name: str
    source_code: str
    complexity: str  # 'simple', 'medium', 'complex'
    category: str    # 'arithmetic', 'vector', 'reduction', etc.
    kernel_obj: Any = None  # Actual kernel object (set after compilation)


class KernelGenerator:
    """Generates varied Warp kernels programmatically."""
    
    def __init__(self, seed: int = None):
        """Initialize generator with optional random seed."""
        if seed is not None:
            random.seed(seed)
        self.generated_count = 0
    
    def _make_unique_name(self, prefix: str) -> str:
        """Generate unique kernel name."""
        self.generated_count += 1
        return f"{prefix}_{self.generated_count:04d}"
    
    def generate_arithmetic_kernel(self, ops: List[str] = None) -> KernelSpec:
        """Generate simple arithmetic kernel."""
        if ops is None:
            ops = random.sample(['+', '-', '*'], k=random.randint(2, 3))
        
        name = self._make_unique_name("arithmetic")
        
        # Build operation chain
        op_chain = "a[i]"
        for i, op in enumerate(ops):
            op_chain = f"({op_chain} {op} b[i])"
        
        source = f'''import warp as wp

@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    i = wp.tid()
    c[i] = {op_chain}
'''
        
        return KernelSpec(
            name=name,
            source_code=source,
            complexity='simple',
            category='arithmetic'
        )
    
    def generate_vector_kernel(self, dimension: int = 3) -> KernelSpec:
        """Generate vector operations kernel."""
        name = self._make_unique_name("vector")
        
        vec_type = f"wp.vec{dimension}"
        
        operations = [
            "pos + vel * dt",
            "wp.normalize(pos + vel * dt)",
        ]
        op = random.choice(operations)
        
        source = f'''import warp as wp

@wp.kernel
def {name}(pos: wp.array(dtype={vec_type}), 
           vel: wp.array(dtype={vec_type}),
           result: wp.array(dtype={vec_type}),
           dt: float):
    i = wp.tid()
    result[i] = {op}
'''
        
        return KernelSpec(
            name=name,
            source_code=source,
            complexity='medium',
            category='vector'
        )
    
    def generate_conditional_kernel(self) -> KernelSpec:
        """Generate kernel with conditional logic."""
        name = self._make_unique_name("conditional")
        
        conditions = [
            ("val < 0.0", "-val", "val * 2.0", "val"),
            ("val < threshold", "val * 2.0", "threshold", "val * 0.5"),
            ("val > 1.0", "1.0", "val", "0.0")
        ]
        cond, true_expr, false_expr, else_expr = random.choice(conditions)
        
        source = f'''import warp as wp

@wp.kernel
def {name}(data: wp.array(dtype=float),
           threshold: float,
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i]
    
    if {cond}:
        output[i] = {true_expr}
    elif val < threshold * 0.5:
        output[i] = {false_expr}
    else:
        output[i] = {else_expr}
'''
        
        return KernelSpec(
            name=name,
            source_code=source,
            complexity='medium',
            category='conditional'
        )
    
    def generate_loop_kernel(self, loop_size: int = None) -> KernelSpec:
        """Generate kernel with loops."""
        if loop_size is None:
            loop_size = random.randint(3, 10)
        
        name = self._make_unique_name("loop")
        
        source = f'''import warp as wp

@wp.kernel
def {name}(matrix: wp.array(dtype=float, ndim=2),
           vector: wp.array(dtype=float),
           result: wp.array(dtype=float),
           n: int):
    i = wp.tid()
    
    sum_val = float(0.0)
    for j in range(n):
        sum_val = sum_val + matrix[i, j] * vector[j]
    
    result[i] = sum_val
'''
        
        return KernelSpec(
            name=name,
            source_code=source,
            complexity='medium',
            category='loop'
        )
    
    def generate_function_kernel(self) -> KernelSpec:
        """Generate kernel with helper functions."""
        name = self._make_unique_name("function")
        func_name = f"helper_{self.generated_count}"
        
        helper_funcs = [
            (f"def {func_name}(x: float) -> float:\n    return x * x + 1.0", "val"),
            (f"def {func_name}(x: float) -> float:\n    return wp.sqrt(wp.abs(x))", "val"),
        ]
        helper_def, arg = random.choice(helper_funcs)
        
        source = f'''import warp as wp

@wp.func
{helper_def}

@wp.kernel
def {name}(data: wp.array(dtype=float),
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i]
    output[i] = {func_name}({arg})
'''
        
        return KernelSpec(
            name=name,
            source_code=source,
            complexity='complex',
            category='function'
        )
    
    def generate_reduction_kernel(self) -> KernelSpec:
        """Generate reduction-style kernel."""
        name = self._make_unique_name("reduction")
        
        source = f'''import warp as wp

@wp.kernel
def {name}(data: wp.array(dtype=float),
           result: wp.array(dtype=float),
           n: int):
    tid = wp.tid()
    
    # Local reduction (simplified for demonstration)
    local_result = float(0.0) if tid < n else float(0.0)
    
    for i in range(tid, n, 1):
        val = data[i]
        local_result = local_result + val * val
        break  # Simplified
    
    result[tid] = local_result
'''
        
        return KernelSpec(
            name=name,
            source_code=source,
            complexity='medium',
            category='reduction'
        )
    
    def generate_math_kernel(self) -> KernelSpec:
        """Generate kernel with math functions."""
        name = self._make_unique_name("math")
        
        math_ops = [
            "wp.sin(val * 3.14159)",
            "wp.exp(val * 0.1)",
            "wp.sqrt(wp.abs(val))",
            "wp.pow(val, 2.0)",
            "wp.log(wp.abs(val) + 1.0)"
        ]
        op = random.choice(math_ops)
        
        source = f'''import warp as wp

@wp.kernel
def {name}(data: wp.array(dtype=float),
           scale: float,
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i] * scale
    output[i] = {op}
'''
        
        return KernelSpec(
            name=name,
            source_code=source,
            complexity='simple',
            category='math'
        )
    
    def generate_random_kernel(self) -> KernelSpec:
        """Generate a random kernel from available types."""
        generators = [
            self.generate_arithmetic_kernel,
            self.generate_vector_kernel,
            self.generate_conditional_kernel,
            self.generate_loop_kernel,
            self.generate_function_kernel,
            self.generate_reduction_kernel,
            self.generate_math_kernel,
        ]
        
        generator = random.choice(generators)
        return generator()
    
    def generate_batch(self, count: int, categories: List[str] = None) -> List[KernelSpec]:
        """Generate a batch of kernels.
        
        Args:
            count: Number of kernels to generate
            categories: List of categories to include, or None for all
            
        Returns:
            List of KernelSpec objects
        """
        kernels = []
        
        for _ in range(count):
            if categories:
                # Generate from specific categories
                category_generators = {
                    'arithmetic': self.generate_arithmetic_kernel,
                    'vector': self.generate_vector_kernel,
                    'conditional': self.generate_conditional_kernel,
                    'loop': self.generate_loop_kernel,
                    'function': self.generate_function_kernel,
                    'reduction': self.generate_reduction_kernel,
                    'math': self.generate_math_kernel,
                }
                cat = random.choice(categories)
                kernel = category_generators[cat]()
            else:
                kernel = self.generate_random_kernel()
            
            kernels.append(kernel)
        
        return kernels


def main():
    """Test kernel generator."""
    print("="*60)
    print("KERNEL GENERATOR TEST")
    print("="*60)
    
    generator = KernelGenerator(seed=42)
    
    # Generate one of each type
    print("\nGenerating test kernels...")
    
    kernels = [
        generator.generate_arithmetic_kernel(),
        generator.generate_vector_kernel(),
        generator.generate_conditional_kernel(),
        generator.generate_loop_kernel(),
        generator.generate_function_kernel(),
        generator.generate_math_kernel(),
    ]
    
    for kernel in kernels:
        print(f"\n{kernel.name} ({kernel.category}, {kernel.complexity}):")
        lines = kernel.source_code.split('\n')
        # Show first few lines
        for line in lines[:8]:
            print(f"  {line}")
        if len(lines) > 8:
            print(f"  ... ({len(lines)-8} more lines)")
    
    print("\n" + "="*60)
    print(f"Generated {len(kernels)} kernel specifications")
    print("Note: Kernels must be written to files to compile")
    print("="*60)


if __name__ == "__main__":
    main()
