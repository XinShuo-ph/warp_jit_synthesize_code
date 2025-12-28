"""Programmatic kernel generator for varied Python→IR pairs."""
import random
import string
from dataclasses import dataclass, field
from typing import List, Optional
import textwrap


@dataclass
class KernelSpec:
    """Specification for a generated kernel."""
    name: str
    params: List[tuple]  # [(name, type_str), ...]
    body_lines: List[str]
    uses_tid: bool = True
    helper_funcs: List[str] = field(default_factory=list)


# Type specifications
SCALAR_TYPES = ["float", "int"]
ARRAY_DTYPES = ["float", "int", "wp.float32", "wp.int32"]
VECTOR_TYPES = ["wp.vec2", "wp.vec3", "wp.vec4"]
MATRIX_TYPES = ["wp.mat22", "wp.mat33"]

# Operations by category
BINARY_OPS = ["+", "-", "*"]
COMPARE_OPS = [">", "<", ">=", "<=", "=="]
MATH_FUNCS = ["wp.sin", "wp.cos", "wp.sqrt", "wp.abs", "wp.exp", "wp.log"]
VEC_FUNCS = ["wp.length", "wp.normalize", "wp.dot", "wp.cross"]


def random_name(prefix: str = "kernel") -> str:
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


class KernelGenerator:
    """Generator for varied warp kernels."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
    
    def gen_arithmetic(self) -> KernelSpec:
        """Generate a simple arithmetic kernel."""
        name = random_name("arith")
        op1 = random.choice(BINARY_OPS)
        op2 = random.choice(BINARY_OPS)
        const = random_float()
        
        params = [
            ("a", "wp.array(dtype=float)"),
            ("b", "wp.array(dtype=float)"),
            ("out", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            f"out[tid] = (a[tid] {op1} b[tid]) {op2} {const}",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_conditional(self) -> KernelSpec:
        """Generate a kernel with conditional logic."""
        name = random_name("cond")
        cmp_op = random.choice(COMPARE_OPS)
        threshold = random_float()
        scale_true = random_float()
        scale_false = random_float()
        
        params = [
            ("x", "wp.array(dtype=float)"),
            ("out", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            f"if x[tid] {cmp_op} {threshold}:",
            f"    out[tid] = x[tid] * {scale_true}",
            "else:",
            f"    out[tid] = x[tid] * {scale_false}",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_loop(self) -> KernelSpec:
        """Generate a kernel with a loop."""
        name = random_name("loop")
        iterations = random.randint(2, 8)
        op = random.choice(BINARY_OPS)
        
        params = [
            ("arr", "wp.array(dtype=float)"),
            ("out", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            "acc = float(0.0)",
            f"for i in range({iterations}):",
            f"    acc = acc {op} arr[tid] * float(i + 1)",
            "out[tid] = acc",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_math_func(self) -> KernelSpec:
        """Generate a kernel using math functions."""
        name = random_name("math")
        func1 = random.choice(MATH_FUNCS)
        func2 = random.choice(MATH_FUNCS)
        scale = random_float()
        
        params = [
            ("x", "wp.array(dtype=float)"),
            ("out", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            f"val = {func1}(x[tid] * {scale})",
            f"out[tid] = {func2}(val)",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_vector(self) -> KernelSpec:
        """Generate a kernel with vector operations."""
        name = random_name("vec")
        dt = random_float()
        
        params = [
            ("pos", "wp.array(dtype=wp.vec3)"),
            ("vel", "wp.array(dtype=wp.vec3)"),
            ("acc", "wp.array(dtype=wp.vec3)"),
        ]
        
        body = [
            "tid = wp.tid()",
            f"dt = {dt}",
            "new_vel = vel[tid] + acc[tid] * dt",
            "pos[tid] = pos[tid] + new_vel * dt",
            "vel[tid] = new_vel",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)

    def gen_matrix(self) -> KernelSpec:
        """Generate a kernel with matrix operations (mat33 * vec3)."""
        name = random_name("mat")

        params = [
            ("A", "wp.array(dtype=wp.mat33)"),
            ("v", "wp.array(dtype=wp.vec3)"),
            ("out", "wp.array(dtype=wp.vec3)"),
        ]

        body = [
            "tid = wp.tid()",
            "out[tid] = A[tid] * v[tid]",
        ]

        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_atomic(self) -> KernelSpec:
        """Generate a kernel with atomic operations."""
        name = random_name("atomic")
        scale = random_float()
        
        params = [
            ("values", "wp.array(dtype=float)"),
            ("result", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            f"wp.atomic_add(result, 0, values[tid] * {scale})",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_nested_loop(self) -> KernelSpec:
        """Generate a kernel with nested loops."""
        name = random_name("nested")
        outer = random.randint(2, 4)
        inner = random.randint(2, 4)
        
        params = [
            ("data", "wp.array(dtype=float)"),
            ("out", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            "total = float(0.0)",
            f"for i in range({outer}):",
            f"    for j in range({inner}):",
            "        total = total + data[tid] * float(i * j + 1)",
            "out[tid] = total",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_multi_conditional(self) -> KernelSpec:
        """Generate a kernel with multiple conditions."""
        name = random_name("multicond")
        t1, t2 = sorted([float(random_float()), float(random_float())])
        
        params = [
            ("x", "wp.array(dtype=float)"),
            ("out", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            "val = x[tid]",
            f"if val < {t1}:",
            "    out[tid] = val * 0.5",
            f"elif val < {t2}:",
            "    out[tid] = val * 1.0",
            "else:",
            "    out[tid] = val * 2.0",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_combined(self) -> KernelSpec:
        """Generate a kernel combining multiple features."""
        name = random_name("combined")
        iterations = random.randint(2, 5)
        threshold = random_float()
        func = random.choice(MATH_FUNCS)
        
        params = [
            ("a", "wp.array(dtype=float)"),
            ("b", "wp.array(dtype=float)"),
            ("out", "wp.array(dtype=float)"),
        ]
        
        body = [
            "tid = wp.tid()",
            "acc = float(0.0)",
            f"for i in range({iterations}):",
            f"    if a[tid] * float(i) > {threshold}:",
            f"        acc = acc + {func}(b[tid])",
            "    else:",
            "        acc = acc + b[tid]",
            "out[tid] = acc",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def gen_with_scalar_param(self) -> KernelSpec:
        """Generate a kernel with scalar parameters."""
        name = random_name("scalar")
        op = random.choice(BINARY_OPS)
        
        params = [
            ("x", "wp.array(dtype=float)"),
            ("out", "wp.array(dtype=float)"),
            ("scale", "float"),
            ("offset", "float"),
        ]
        
        body = [
            "tid = wp.tid()",
            f"out[tid] = x[tid] {op} scale + offset",
        ]
        
        return KernelSpec(name=name, params=params, body_lines=body)
    
    def generate(self, kernel_type: Optional[str] = None) -> KernelSpec:
        """Generate a kernel of the specified type or random."""
        generators = {
            "arithmetic": self.gen_arithmetic,
            "conditional": self.gen_conditional,
            "loop": self.gen_loop,
            "math": self.gen_math_func,
            "vector": self.gen_vector,
            "matrix": self.gen_matrix,
            "atomic": self.gen_atomic,
            "nested": self.gen_nested_loop,
            "multi_cond": self.gen_multi_conditional,
            "combined": self.gen_combined,
            "scalar_param": self.gen_with_scalar_param,
        }
        
        if kernel_type is None:
            kernel_type = random.choice(list(generators.keys()))
        
        return generators[kernel_type]()
    
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


def generate_kernel_module(specs: List[KernelSpec], module_name: str) -> str:
    """Generate a complete Python module with multiple kernels."""
    gen = KernelGenerator()
    
    header = '''"""Auto-generated warp kernels for training data synthesis."""
import warp as wp

wp.init()

'''
    
    kernels = "\n\n".join(gen.to_python_source(spec) for spec in specs)
    
    return header + kernels


if __name__ == "__main__":
    # Demo: generate 10 different kernel types
    gen = KernelGenerator(seed=42)
    
    print("=== Generated Kernel Examples ===\n")
    
    for kernel_type in ["arithmetic", "conditional", "loop", "math", "vector", "matrix",
                        "atomic", "nested", "multi_cond", "combined", "scalar_param"]:
        spec = gen.generate(kernel_type)
        source = gen.to_python_source(spec)
        print(f"--- {kernel_type} ---")
        print(source)
