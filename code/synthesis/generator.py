#!/usr/bin/env python3
"""
Kernel Generator for Warp - File-based approach

Generates varied Python kernel functions by writing to temporary files
and importing them dynamically.
"""

import warp as wp
import random
import os
import tempfile
import importlib.util
from typing import List, Dict, Any, Callable, Tuple

class KernelGenerator:
    """
    Generates varied warp kernel functions.
    
    Uses temporary Python files to avoid exec() restrictions.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.generated_count = 0
        self.temp_dir = tempfile.mkdtemp(prefix="warp_kernels_")
        
    def _unique_name(self, prefix: str = "kernel") -> str:
        """Generate unique kernel name."""
        self.generated_count += 1
        return f"{prefix}_{self.generated_count}"
    
    def _load_kernel_from_source(self, source: str, kernel_name: str) -> Callable:
        """
        Load a kernel from source code by writing to temp file.
        
        Args:
            source: Python source code containing the kernel
            kernel_name: Name of the kernel function
            
        Returns:
            The kernel function
        """
        # Create temp file
        filename = f"temp_kernel_{self.generated_count}.py"
        filepath = os.path.join(self.temp_dir, filename)
        
        # Write source
        with open(filepath, 'w') as f:
            f.write("import warp as wp\n\n")
            f.write(source)
        
        # Load module
        spec = importlib.util.spec_from_file_location(
            f"temp_module_{self.generated_count}", 
            filepath
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get kernel function
        kernel_func = getattr(module, kernel_name)
        
        return kernel_func
    
    def generate_simple_map(self) -> Tuple[Callable, Dict[str, Any], str]:
        """
        Generate simple element-wise map operation.
        Pattern: out[i] = f(in[i])
        
        Returns:
            (kernel_func, params, source)
        """
        name = self._unique_name("map")
        
        # Choose operation
        arithmetic_ops = ['+', '-', '*']
        op1 = self.rng.choice(arithmetic_ops)
        op2 = self.rng.choice(arithmetic_ops)
        const1 = self.rng.uniform(0.5, 5.0)
        const2 = self.rng.uniform(0.5, 5.0)
        
        source = f"""@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    tid = wp.tid()
    x = input[tid]
    y = x {op1} {const1:.2f}
    z = y {op2} {const2:.2f}
    output[tid] = z
"""
        
        kernel_func = self._load_kernel_from_source(source, name)
        
        # Generate test parameters
        n = self.rng.randint(10, 50)
        input_data = [float(self.rng.random() * 10.0) for _ in range(n)]
        
        params = {
            'launch_args': {
                'dim': n,
                'inputs': [
                    wp.array(input_data, dtype=float),
                    wp.zeros(n, dtype=float)
                ]
            }
        }
        
        return kernel_func, params, source
    
    def generate_reduce_sum(self) -> Tuple[Callable, Dict[str, Any], str]:
        """Generate reduction kernel."""
        name = self._unique_name("reduce")
        
        source = f"""@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(output, 0, input[tid])
"""
        
        kernel_func = self._load_kernel_from_source(source, name)
        
        n = self.rng.randint(10, 50)
        input_data = [float(self.rng.random()) for _ in range(n)]
        
        params = {
            'launch_args': {
                'dim': n,
                'inputs': [
                    wp.array(input_data, dtype=float),
                    wp.zeros(1, dtype=float)
                ]
            }
        }
        
        return kernel_func, params, source
    
    def generate_conditional(self) -> Tuple[Callable, Dict[str, Any], str]:
        """Generate kernel with conditional logic."""
        name = self._unique_name("cond")
        
        comp_ops = ['<', '>', '<=', '>=']
        comp_op = self.rng.choice(comp_ops)
        threshold = self.rng.uniform(0.0, 5.0)
        
        op1_val = self.rng.uniform(1.0, 3.0)
        op2_val = self.rng.uniform(1.0, 3.0)
        
        source = f"""@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    tid = wp.tid()
    x = input[tid]
    
    if x {comp_op} {threshold:.2f}:
        output[tid] = x * {op1_val:.2f}
    else:
        output[tid] = x + {op2_val:.2f}
"""
        
        kernel_func = self._load_kernel_from_source(source, name)
        
        n = self.rng.randint(10, 50)
        input_data = [float(self.rng.uniform(0.0, 10.0)) for _ in range(n)]
        
        params = {
            'launch_args': {
                'dim': n,
                'inputs': [
                    wp.array(input_data, dtype=float),
                    wp.zeros(n, dtype=float)
                ]
            }
        }
        
        return kernel_func, params, source
    
    def generate_math_func(self) -> Tuple[Callable, Dict[str, Any], str]:
        """Generate kernel with math functions."""
        name = self._unique_name("math")
        
        funcs = ['wp.sin', 'wp.cos', 'wp.abs', 'wp.sqrt']
        func1 = self.rng.choice(funcs)
        func2 = self.rng.choice(funcs)
        scale = self.rng.uniform(0.1, 2.0)
        
        source = f"""@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    tid = wp.tid()
    x = input[tid]
    y = {func1}(x * {scale:.2f})
    z = {func2}(y)
    output[tid] = z
"""
        
        kernel_func = self._load_kernel_from_source(source, name)
        
        n = self.rng.randint(10, 50)
        input_data = [float(self.rng.uniform(0.1, 10.0)) for _ in range(n)]
        
        params = {
            'launch_args': {
                'dim': n,
                'inputs': [
                    wp.array(input_data, dtype=float),
                    wp.zeros(n, dtype=float)
                ]
            }
        }
        
        return kernel_func, params, source
    
    def generate_vector_dot(self) -> Tuple[Callable, Dict[str, Any], str]:
        """Generate vector dot product kernel."""
        name = self._unique_name("vec_dot")
        
        source = f"""@wp.kernel
def {name}(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), 
           output: wp.array(dtype=float)):
    tid = wp.tid()
    output[tid] = wp.dot(a[tid], b[tid])
"""
        
        kernel_func = self._load_kernel_from_source(source, name)
        
        n = self.rng.randint(10, 30)
        vecs_a = [[1.0, 0.0, 0.0] for _ in range(n)]
        vecs_b = [[0.0, 1.0, 0.0] for _ in range(n)]
        
        params = {
            'launch_args': {
                'dim': n,
                'inputs': [
                    wp.array(vecs_a, dtype=wp.vec3),
                    wp.array(vecs_b, dtype=wp.vec3),
                    wp.zeros(n, dtype=float)
                ]
            }
        }
        
        return kernel_func, params, source
    
    def generate_random(self) -> Tuple[Callable, Dict[str, Any], str]:
        """Generate a random kernel from available templates."""
        templates = [
            self.generate_simple_map,
            self.generate_reduce_sum,
            self.generate_conditional,
            self.generate_math_func,
            self.generate_vector_dot,
        ]
        
        template_func = self.rng.choice(templates)
        return template_func()


def test_generator():
    """Test the kernel generator."""
    print("="*60)
    print("Testing Kernel Generator")
    print("="*60)
    
    gen = KernelGenerator(seed=42)
    
    # Generate several random kernels
    for i in range(5):
        print(f"\nKernel {i+1}:")
        try:
            kernel, params, source = gen.generate_random()
            print(f"  ✓ Generated: {kernel.key}")
            print(f"  Source preview:")
            lines = source.strip().split('\n')
            for line in lines[:4]:
                print(f"    {line}")
            if len(lines) > 4:
                print(f"    ... ({len(lines)} lines total)")
            
            # Test launch
            wp.launch(kernel=kernel, **params['launch_args'])
            wp.synchronize()
            print(f"  ✓ Kernel executed successfully")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Generator test complete!")


if __name__ == "__main__":
    test_generator()
