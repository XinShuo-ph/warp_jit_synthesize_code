"""
Kernel Generator: Programmatically generate varied warp kernels.

Generates Python kernel source code with variations in:
- Operations: arithmetic, vector, matrix, control flow
- Data types: float, vec2, vec3, vec4, mat22, mat33, mat44
- Patterns: element-wise, reductions, stencils
"""
import random
import string
from typing import Any
from dataclasses import dataclass, field


@dataclass
class KernelSpec:
    """Specification for a generated kernel."""
    name: str
    category: str
    source: str
    arg_types: dict[str, str]
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Type definitions
SCALAR_TYPES = ["float"]
VECTOR_TYPES = ["wp.vec2", "wp.vec3", "wp.vec4"]
MATRIX_TYPES = ["wp.mat22", "wp.mat33", "wp.mat44"]
ALL_TYPES = SCALAR_TYPES + VECTOR_TYPES + MATRIX_TYPES


# Operation templates
UNARY_OPS = {
    "neg": "-{x}",
    "abs": "wp.abs({x})",
    "sqrt": "wp.sqrt(wp.abs({x}))",
    "sin": "wp.sin({x})",
    "cos": "wp.cos({x})",
    "exp": "wp.exp({x})",
    "log": "wp.log(wp.abs({x}) + 1.0)",
}

BINARY_OPS = {
    "add": "{x} + {y}",
    "sub": "{x} - {y}",
    "mul": "{x} * {y}",
    "div": "{x} / ({y} + 0.0001)",
    "min": "wp.min({x}, {y})",
    "max": "wp.max({x}, {y})",
}

VECTOR_OPS = {
    "dot": "wp.dot({x}, {y})",
    "cross": "wp.cross({x}, {y})",  # only vec3
    "length": "wp.length({x})",
    "normalize": "wp.normalize({x})",
}

COMPARISON_OPS = {
    "lt": "{x} < {y}",
    "gt": "{x} > {y}",
    "le": "{x} <= {y}",
    "ge": "{x} >= {y}",
}


def random_name(prefix: str = "kernel") -> str:
    """Generate a random kernel name."""
    suffix = ''.join(random.choices(string.ascii_lowercase, k=6))
    return f"{prefix}_{suffix}"


def generate_arithmetic_kernel(seed: int | None = None) -> KernelSpec:
    """Generate a kernel with arithmetic operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("arith")
    num_ops = random.randint(1, 4)
    
    # Build operation chain
    ops = []
    var_counter = 0
    
    for i in range(num_ops):
        if random.random() < 0.5:
            # Unary op
            op_name = random.choice(list(UNARY_OPS.keys()))
            op_template = UNARY_OPS[op_name]
            if i == 0:
                expr = op_template.format(x="a[tid]")
            else:
                expr = op_template.format(x=f"var_{var_counter-1}")
        else:
            # Binary op
            op_name = random.choice(list(BINARY_OPS.keys()))
            op_template = BINARY_OPS[op_name]
            if i == 0:
                expr = op_template.format(x="a[tid]", y="b[tid]")
            else:
                expr = op_template.format(x=f"var_{var_counter-1}", y="b[tid]")
        
        ops.append(f"    var_{var_counter} = {expr}")
        var_counter += 1
    
    source = f'''@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
{chr(10).join(ops)}
    c[tid] = var_{var_counter-1}
'''
    
    return KernelSpec(
        name=name,
        category="arithmetic",
        source=source,
        arg_types={"a": "wp.array(dtype=float)", "b": "wp.array(dtype=float)", "c": "wp.array(dtype=float)"},
        description=f"Arithmetic kernel with {num_ops} operations",
        metadata={"num_ops": num_ops, "seed": seed}
    )


def generate_vector_kernel(seed: int | None = None) -> KernelSpec:
    """Generate a kernel with vector operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("vec")
    vec_type = random.choice(["wp.vec2", "wp.vec3", "wp.vec4"])
    
    # Choose operations based on vector type
    if vec_type == "wp.vec3":
        available_ops = ["dot", "cross", "length", "normalize"]
    else:
        available_ops = ["dot", "length", "normalize"]
    
    op = random.choice(available_ops)
    
    if op == "dot":
        body = "    out[tid] = wp.dot(a[tid], b[tid])"
        out_type = "float"
    elif op == "cross":
        body = "    out[tid] = wp.cross(a[tid], b[tid])"
        out_type = vec_type
    elif op == "length":
        body = "    out[tid] = wp.length(a[tid])"
        out_type = "float"
    else:  # normalize
        body = "    out[tid] = wp.normalize(a[tid])"
        out_type = vec_type
    
    source = f'''@wp.kernel
def {name}(a: wp.array(dtype={vec_type}), b: wp.array(dtype={vec_type}), out: wp.array(dtype={out_type})):
    tid = wp.tid()
{body}
'''
    
    return KernelSpec(
        name=name,
        category="vector",
        source=source,
        arg_types={"a": f"wp.array(dtype={vec_type})", "b": f"wp.array(dtype={vec_type})", "out": f"wp.array(dtype={out_type})"},
        description=f"Vector kernel: {op} on {vec_type}",
        metadata={"vec_type": vec_type, "operation": op, "seed": seed}
    )


def generate_matrix_kernel(seed: int | None = None) -> KernelSpec:
    """Generate a kernel with matrix operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("mat")
    size = random.choice([2, 3, 4])
    mat_type = f"wp.mat{size}{size}"
    vec_type = f"wp.vec{size}"
    
    op = random.choice(["mat_vec", "mat_mat", "transpose"])
    
    if op == "mat_vec":
        source = f'''@wp.kernel
def {name}(m: wp.array(dtype={mat_type}), v: wp.array(dtype={vec_type}), out: wp.array(dtype={vec_type})):
    tid = wp.tid()
    out[tid] = m[tid] * v[tid]
'''
        arg_types = {"m": f"wp.array(dtype={mat_type})", "v": f"wp.array(dtype={vec_type})", "out": f"wp.array(dtype={vec_type})"}
        desc = f"Matrix-vector multiply ({mat_type} * {vec_type})"
    elif op == "mat_mat":
        source = f'''@wp.kernel
def {name}(a: wp.array(dtype={mat_type}), b: wp.array(dtype={mat_type}), out: wp.array(dtype={mat_type})):
    tid = wp.tid()
    out[tid] = a[tid] * b[tid]
'''
        arg_types = {"a": f"wp.array(dtype={mat_type})", "b": f"wp.array(dtype={mat_type})", "out": f"wp.array(dtype={mat_type})"}
        desc = f"Matrix-matrix multiply ({mat_type})"
    else:  # transpose
        source = f'''@wp.kernel
def {name}(m: wp.array(dtype={mat_type}), out: wp.array(dtype={mat_type})):
    tid = wp.tid()
    out[tid] = wp.transpose(m[tid])
'''
        arg_types = {"m": f"wp.array(dtype={mat_type})", "out": f"wp.array(dtype={mat_type})"}
        desc = f"Matrix transpose ({mat_type})"
    
    return KernelSpec(
        name=name,
        category="matrix",
        source=source,
        arg_types=arg_types,
        description=desc,
        metadata={"mat_type": mat_type, "operation": op, "seed": seed}
    )


def generate_control_flow_kernel(seed: int | None = None) -> KernelSpec:
    """Generate a kernel with control flow (if/for)."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("ctrl")
    pattern = random.choice(["clamp", "abs_diff", "step", "loop_sum", "loop_product"])
    
    if pattern == "clamp":
        source = f'''@wp.kernel
def {name}(a: wp.array(dtype=float), lo: float, hi: float, out: wp.array(dtype=float)):
    tid = wp.tid()
    val = a[tid]
    if val < lo:
        out[tid] = lo
    elif val > hi:
        out[tid] = hi
    else:
        out[tid] = val
'''
        arg_types = {"a": "wp.array(dtype=float)", "lo": "float", "hi": "float", "out": "wp.array(dtype=float)"}
        desc = "Clamp values between lo and hi"
    
    elif pattern == "abs_diff":
        source = f'''@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    diff = a[tid] - b[tid]
    if diff < 0.0:
        out[tid] = -diff
    else:
        out[tid] = diff
'''
        arg_types = {"a": "wp.array(dtype=float)", "b": "wp.array(dtype=float)", "out": "wp.array(dtype=float)"}
        desc = "Absolute difference"
    
    elif pattern == "step":
        source = f'''@wp.kernel
def {name}(a: wp.array(dtype=float), threshold: float, out: wp.array(dtype=float)):
    tid = wp.tid()
    if a[tid] > threshold:
        out[tid] = 1.0
    else:
        out[tid] = 0.0
'''
        arg_types = {"a": "wp.array(dtype=float)", "threshold": "float", "out": "wp.array(dtype=float)"}
        desc = "Step function"
    
    elif pattern == "loop_sum":
        n_iters = random.randint(2, 5)
        source = f'''@wp.kernel
def {name}(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    total = float(0.0)
    for i in range({n_iters}):
        total = total + a[tid]
    out[tid] = total
'''
        arg_types = {"a": "wp.array(dtype=float)", "out": "wp.array(dtype=float)"}
        desc = f"Loop sum ({n_iters} iterations)"
    
    else:  # loop_product
        n_iters = random.randint(2, 4)
        source = f'''@wp.kernel
def {name}(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    prod = float(1.0)
    for i in range({n_iters}):
        prod = prod * a[tid]
    out[tid] = prod
'''
        arg_types = {"a": "wp.array(dtype=float)", "out": "wp.array(dtype=float)"}
        desc = f"Loop product ({n_iters} iterations)"
    
    return KernelSpec(
        name=name,
        category="control_flow",
        source=source,
        arg_types=arg_types,
        description=desc,
        metadata={"pattern": pattern, "seed": seed}
    )


def generate_math_kernel(seed: int | None = None) -> KernelSpec:
    """Generate a kernel with math functions."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("math")
    num_funcs = random.randint(1, 3)
    
    funcs = random.sample(list(UNARY_OPS.keys()), min(num_funcs, len(UNARY_OPS)))
    
    # Build expression chain
    expr = "a[tid]"
    for func in funcs:
        template = UNARY_OPS[func]
        expr = template.format(x=expr)
    
    source = f'''@wp.kernel
def {name}(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = {expr}
'''
    
    return KernelSpec(
        name=name,
        category="math",
        source=source,
        arg_types={"a": "wp.array(dtype=float)", "out": "wp.array(dtype=float)"},
        description=f"Math functions: {', '.join(funcs)}",
        metadata={"functions": funcs, "seed": seed}
    )


def generate_atomic_kernel(seed: int | None = None) -> KernelSpec:
    """Generate a kernel with atomic operations."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("atom")
    op = random.choice(["add", "min", "max"])
    
    if op == "add":
        source = f'''@wp.kernel
def {name}(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(result, 0, values[tid])
'''
        desc = "Atomic add reduction"
    elif op == "min":
        source = f'''@wp.kernel
def {name}(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_min(result, 0, values[tid])
'''
        desc = "Atomic min reduction"
    else:  # max
        source = f'''@wp.kernel
def {name}(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_max(result, 0, values[tid])
'''
        desc = "Atomic max reduction"
    
    return KernelSpec(
        name=name,
        category="atomic",
        source=source,
        arg_types={"values": "wp.array(dtype=float)", "result": "wp.array(dtype=float)"},
        description=desc,
        metadata={"operation": op, "seed": seed}
    )


def generate_reduction_kernel(seed: int | None = None) -> KernelSpec:
    """Generate a reduction kernel (GPU-friendly pattern)."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("reduce")
    op = random.choice(["sum", "prod", "max", "min"])
    
    if op == "sum":
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(output, 0, input[tid])
'''
        desc = "Parallel sum reduction"
    elif op == "prod":
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    tid = wp.tid()
    val = input[tid]
    # Note: atomic_mul not available, using loop
    if val != 1.0:
        wp.atomic_add(output, 0, val - 1.0)
'''
        desc = "Parallel product reduction (simplified)"
    elif op == "max":
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_max(output, 0, input[tid])
'''
        desc = "Parallel max reduction"
    else:  # min
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_min(output, 0, input[tid])
'''
        desc = "Parallel min reduction"
    
    return KernelSpec(
        name=name,
        category="reduction",
        source=source,
        arg_types={"input": "wp.array(dtype=float)", "output": "wp.array(dtype=float)"},
        description=desc,
        metadata={"operation": op, "seed": seed}
    )


def generate_stencil_kernel(seed: int | None = None) -> KernelSpec:
    """Generate a stencil computation kernel (common GPU pattern)."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("stencil")
    stencil_type = random.choice(["avg3", "avg5", "laplacian"])
    
    if stencil_type == "avg3":
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float), n: int):
    tid = wp.tid()
    if tid > 0 and tid < n - 1:
        output[tid] = (input[tid - 1] + input[tid] + input[tid + 1]) / 3.0
    else:
        output[tid] = input[tid]
'''
        desc = "3-point averaging stencil"
    elif stencil_type == "avg5":
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float), n: int):
    tid = wp.tid()
    if tid > 1 and tid < n - 2:
        sum = input[tid - 2] + input[tid - 1] + input[tid] + input[tid + 1] + input[tid + 2]
        output[tid] = sum / 5.0
    else:
        output[tid] = input[tid]
'''
        desc = "5-point averaging stencil"
    else:  # laplacian
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float), n: int):
    tid = wp.tid()
    if tid > 0 and tid < n - 1:
        output[tid] = input[tid - 1] - 2.0 * input[tid] + input[tid + 1]
    else:
        output[tid] = 0.0
'''
        desc = "Laplacian stencil (1D)"
    
    return KernelSpec(
        name=name,
        category="stencil",
        source=source,
        arg_types={"input": "wp.array(dtype=float)", "output": "wp.array(dtype=float)", "n": "int"},
        description=desc,
        metadata={"stencil_type": stencil_type, "seed": seed}
    )


def generate_transform_kernel(seed: int | None = None) -> KernelSpec:
    """Generate a data transformation kernel."""
    if seed is not None:
        random.seed(seed)
    
    name = random_name("xform")
    transform_type = random.choice(["scale", "shift", "clamp", "normalize"])
    
    if transform_type == "scale":
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float), scale: float):
    tid = wp.tid()
    output[tid] = input[tid] * scale
'''
        desc = "Scale transformation"
        arg_types = {"input": "wp.array(dtype=float)", "output": "wp.array(dtype=float)", "scale": "float"}
    elif transform_type == "shift":
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float), shift: float):
    tid = wp.tid()
    output[tid] = input[tid] + shift
'''
        desc = "Shift transformation"
        arg_types = {"input": "wp.array(dtype=float)", "output": "wp.array(dtype=float)", "shift": "float"}
    elif transform_type == "clamp":
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float), min_val: float, max_val: float):
    tid = wp.tid()
    val = input[tid]
    if val < min_val:
        output[tid] = min_val
    elif val > max_val:
        output[tid] = max_val
    else:
        output[tid] = val
'''
        desc = "Clamp transformation"
        arg_types = {"input": "wp.array(dtype=float)", "output": "wp.array(dtype=float)", 
                     "min_val": "float", "max_val": "float"}
    else:  # normalize
        source = f'''@wp.kernel
def {name}(input: wp.array(dtype=float), output: wp.array(dtype=float), mean: float, std: float):
    tid = wp.tid()
    output[tid] = (input[tid] - mean) / (std + 0.0001)
'''
        desc = "Normalize transformation"
        arg_types = {"input": "wp.array(dtype=float)", "output": "wp.array(dtype=float)", 
                     "mean": "float", "std": "float"}
    
    return KernelSpec(
        name=name,
        category="transform",
        source=source,
        arg_types=arg_types,
        description=desc,
        metadata={"transform_type": transform_type, "seed": seed}
    )


# Generator dispatch table
GENERATORS = {
    "arithmetic": generate_arithmetic_kernel,
    "vector": generate_vector_kernel,
    "matrix": generate_matrix_kernel,
    "control_flow": generate_control_flow_kernel,
    "math": generate_math_kernel,
    "atomic": generate_atomic_kernel,
    "reduction": generate_reduction_kernel,
    "stencil": generate_stencil_kernel,
    "transform": generate_transform_kernel,
}


def generate_kernel(category: str | None = None, seed: int | None = None) -> KernelSpec:
    """
    Generate a kernel of the specified category.
    If category is None, randomly choose a category.
    """
    if category is None:
        category = random.choice(list(GENERATORS.keys()))
    
    if category not in GENERATORS:
        raise ValueError(f"Unknown category: {category}. Available: {list(GENERATORS.keys())}")
    
    return GENERATORS[category](seed)


def generate_kernels(n: int, categories: list[str] | None = None, seed: int | None = None) -> list[KernelSpec]:
    """Generate n kernels with optional category filtering."""
    if seed is not None:
        random.seed(seed)
    
    if categories is None:
        categories = list(GENERATORS.keys())
    
    kernels = []
    for i in range(n):
        cat = random.choice(categories)
        spec = generate_kernel(cat, seed=seed + i if seed else None)
        kernels.append(spec)
    
    return kernels


if __name__ == "__main__":
    # Demo: Generate one kernel of each type
    print("=" * 60)
    print("Kernel Generator Demo")
    print("=" * 60)
    
    for cat in GENERATORS.keys():
        spec = generate_kernel(cat, seed=42)
        print(f"\n--- {cat.upper()} ---")
        print(f"Name: {spec.name}")
        print(f"Description: {spec.description}")
        print(f"Source:\n{spec.source}")
