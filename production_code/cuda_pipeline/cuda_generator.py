"""
CUDA-Adapted Kernel Generator for Python→CUDA IR Pairs
Generates kernels specifically designed for CUDA backend compilation.
"""
import random
import string
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KernelSpec:
    """Specification for a generated kernel."""
    name: str
    params: List[tuple]  # [(name, type_str), ...]
    body_lines: List[str]
    uses_tid: bool = True
    helper_funcs: List[str] = field(default_factory=list)
    source: str = ""  # Python source code for the kernel


# CUDA-specific types
CUDA_SCALAR_TYPES = ["float", "int"]
CUDA_ARRAY_DTYPES = ["float", "int", "wp.float32", "wp.int32"]
CUDA_VECTOR_TYPES = ["wp.vec2", "wp.vec3", "wp.vec4"]
CUDA_MATRIX_TYPES = ["wp.mat22", "wp.mat33", "wp.mat44"]

# CUDA operations
BINARY_OPS = ["+", "-", "*", "/"]
COMPARE_OPS = [">", "<", ">=", "<=", "==", "!="]
MATH_FUNCS = ["wp.sin", "wp.cos", "wp.sqrt", "wp.abs", "wp.exp", "wp.log", "wp.tan"]
VEC_FUNCS = ["wp.length", "wp.normalize", "wp.dot", "wp.cross"]
ATOMIC_OPS = ["wp.atomic_add", "wp.atomic_sub", "wp.atomic_max", "wp.atomic_min"]


def random_name(prefix: str = "cuda_kernel") -> str:
    """Generate a random kernel name."""
    suffix = ''.join(random.choices(string.ascii_lowercase, k=6))
    return f"{prefix}_{suffix}"


def random_float() -> str:
    """Generate a random float literal."""
    val = round(random.uniform(-10, 10), 2)
    return f"{val}"


def random_int() -> str:
    """Generate a random int literal."""
    return str(random.randint(1, 100))


class CUDAKernelGenerator:
    """Generator for CUDA-targeted warp kernels."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
    
    def gen_cuda_arithmetic(self) -> KernelSpec:
        """Generate CUDA arithmetic kernel with thread indexing."""
        name = random_name("cuda_arith")
        op1 = random.choice(BINARY_OPS)
        op2 = random.choice(BINARY_OPS)
        const = random_float()
        
        params = [
            ("a", "wp.array(dtype=float)"),
            ("b", "wp.array(dtype=float)"),
            ("c", "wp.array(dtype=float)"),
            ("out", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            f"val = a[tid] {op1} b[tid]",
            f"out[tid] = val {op2} c[tid] {op1} {const}",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_cuda_reduction(self) -> KernelSpec:
        """Generate CUDA reduction kernel with atomics."""
        name = random_name("cuda_reduce")
        scale = random_float()
        op = random.choice(ATOMIC_OPS)
        
        params = [
            ("values", "wp.array(dtype=float)"),
            ("result", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            f"val = values[tid] * {scale}",
            f"{op}(result, 0, val)",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_cuda_stencil(self) -> KernelSpec:
        """Generate CUDA stencil computation."""
        name = random_name("cuda_stencil")
        weight = random_float()
        
        params = [
            ("input", "wp.array(dtype=float)"),
            ("output", "wp.array(dtype=float)"),
            ("n", "int"),
        ]
        
        body = [
            "tid = wp.tid()",
            "if tid > 0 and tid < n - 1:",
            f"    left = input[tid - 1] * {weight}",
            "    center = input[tid]",
            f"    right = input[tid + 1] * {weight}",
            "    output[tid] = left + center + right",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_cuda_vector_ops(self) -> KernelSpec:
        """Generate CUDA vector operations."""
        name = random_name("cuda_vec")
        dt = random_float()
        scale = random_float()
        
        params = [
            ("pos", "wp.array(dtype=wp.vec3)"),
            ("vel", "wp.array(dtype=wp.vec3)"),
            ("acc", "wp.array(dtype=wp.vec3)"),
            ("output", "wp.array(dtype=wp.vec3)"),
        ]
        
        body = [
            "tid = wp.tid()",
            f"dt = {dt}",
            f"new_vel = vel[tid] + acc[tid] * dt * {scale}",
            "new_pos = pos[tid] + new_vel * dt",
            "output[tid] = new_pos",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_cuda_parallel_scan(self) -> KernelSpec:
        """Generate CUDA parallel prefix scan pattern."""
        name = random_name("cuda_scan")
        iterations = random.randint(3, 6)
        
        params = [
            ("data", "wp.array(dtype=float)"),
            ("out", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            "acc = data[tid]",
            f"for i in range({iterations}):",
            "    acc = acc + data[tid] * float(i + 1)",
            "out[tid] = acc",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_cuda_matrix_mul(self) -> KernelSpec:
        """Generate CUDA matrix-vector multiply."""
        name = random_name("cuda_matmul")
        
        params = [
            ("mat", "wp.array(dtype=wp.mat33)"),
            ("vec", "wp.array(dtype=wp.vec3)"),
            ("out", "wp.array(dtype=wp.vec3)"),
        ]
        
        body = [
            "tid = wp.tid()",
            "m = mat[tid]",
            "v = vec[tid]",
            "out[tid] = m * v",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_cuda_conditional_reduce(self) -> KernelSpec:
        """Generate CUDA conditional reduction."""
        name = random_name("cuda_condreduce")
        threshold = random_float()
        scale_a = random_float()
        scale_b = random_float()
        
        params = [
            ("values", "wp.array(dtype=float)"),
            ("flags", "wp.array(dtype=int)"),
            ("result_a", "wp.array(dtype=float)"),
            ("result_b", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            "val = values[tid]",
            f"if val > {threshold}:",
            f"    wp.atomic_add(result_a, 0, val * {scale_a})",
            "else:",
            f"    wp.atomic_add(result_b, 0, val * {scale_b})",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_cuda_warp_ops(self) -> KernelSpec:
        """Generate CUDA with multiple math operations."""
        name = random_name("cuda_warpmath")
        func1 = random.choice(MATH_FUNCS)
        func2 = random.choice(MATH_FUNCS)
        scale = random_float()
        
        params = [
            ("x", "wp.array(dtype=float)"),
            ("y", "wp.array(dtype=float)"),
            ("out", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            f"v1 = {func1}(x[tid] * {scale})",
            f"v2 = {func2}(y[tid])",
            "out[tid] = v1 + v2",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_cuda_nested_loop(self) -> KernelSpec:
        """Generate CUDA kernel with nested loops."""
        name = random_name("cuda_nested")
        outer = random.randint(2, 4)
        inner = random.randint(2, 4)
        
        params = [
            ("data", "wp.array(dtype=float)"),
            ("coeffs", "wp.array(dtype=float)"),
            ("out", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            "total = float(0.0)",
            f"for i in range({outer}):",
            f"    for j in range({inner}):",
            "        idx = i * j",
            "        total = total + data[tid] * coeffs[min(idx, tid)]",
            "out[tid] = total",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_cuda_combined(self) -> KernelSpec:
        """Generate complex CUDA kernel combining multiple patterns."""
        name = random_name("cuda_complex")
        threshold = random_float()
        func = random.choice(MATH_FUNCS)
        iterations = random.randint(3, 5)
        
        params = [
            ("a", "wp.array(dtype=float)"),
            ("b", "wp.array(dtype=float)"),
            ("flags", "wp.array(dtype=int)"),
            ("out", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            "acc = float(0.0)",
            f"for i in range({iterations}):",
            f"    val = a[tid] * float(i + 1)",
            f"    if val > {threshold}:",
            f"        acc = acc + {func}(b[tid])",
            "    else:",
            "        acc = acc + b[tid]",
            "    if flags[tid] > 0:",
            "        acc = acc * 2.0",
            "out[tid] = acc",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def generate(self, kernel_type: Optional[str] = None) -> KernelSpec:
        """Generate a CUDA kernel of the specified type or random."""
        generators = {
            "cuda_arithmetic": self.gen_cuda_arithmetic,
            "cuda_reduction": self.gen_cuda_reduction,
            "cuda_stencil": self.gen_cuda_stencil,
            "cuda_vector": self.gen_cuda_vector_ops,
            "cuda_scan": self.gen_cuda_parallel_scan,
            "cuda_matmul": self.gen_cuda_matrix_mul,
            "cuda_condreduce": self.gen_cuda_conditional_reduce,
            "cuda_warpmath": self.gen_cuda_warp_ops,
            "cuda_nested": self.gen_cuda_nested_loop,
            "cuda_combined": self.gen_cuda_combined,
        }
        
        if kernel_type is None:
            kernel_type = random.choice(list(generators.keys()))
        
        spec = generators[kernel_type]()
        # Generate source code and store it
        spec.source = self.to_python_source(spec)
        return spec
    
    def to_python_source(self, spec: KernelSpec) -> str:
        """Convert a kernel spec to Python source code."""
        # Build parameter string
        param_str = ", ".join(f"{p[0]}: {p[1]}" for p in spec.params)
        
        # Build body with proper indentation
        body = "\n".join("    " + line for line in spec.body_lines)
        
        # Combine into kernel definition
        source = f"""@wp.kernel
def {spec.name}({param_str}):
{body}
"""
        return source


# Export for compatibility
GENERATORS = {
    "cuda_arithmetic": "cuda_arithmetic",
    "cuda_reduction": "cuda_reduction",
    "cuda_stencil": "cuda_stencil",
    "cuda_vector": "cuda_vector",
    "cuda_scan": "cuda_scan",
    "cuda_matmul": "cuda_matmul",
    "cuda_condreduce": "cuda_condreduce",
    "cuda_warpmath": "cuda_warpmath",
    "cuda_nested": "cuda_nested",
    "cuda_combined": "cuda_combined",
}


def generate_kernel(kernel_type: Optional[str] = None, seed: Optional[int] = None) -> KernelSpec:
    """Generate a CUDA kernel (convenience function)."""
    gen = CUDAKernelGenerator(seed=seed)
    return gen.generate(kernel_type)


if __name__ == "__main__":
    # Demo: generate CUDA kernel examples
    gen = CUDAKernelGenerator(seed=42)
    
    print("=== Generated CUDA Kernel Examples ===\n")
    
    for kernel_type in list(GENERATORS.keys()):
        spec = gen.generate(kernel_type)
        print(f"--- {kernel_type} ---")
        print(spec.source)
