"""
Kernel Generator: Programmatically generate varied Warp kernels.

This generator returns Python source strings for @wp.kernel functions.
Downstream pipeline code is responsible for writing/importing this source
into a module that already does `import warp as wp`.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KernelSpec:
    """Specification for a generated kernel."""

    name: str
    category: str
    source: str
    arg_types: dict[str, str]
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Operation templates (Warp intrinsics)
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


def _random_name(prefix: str) -> str:
    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    return f"{prefix}_{suffix}"


def generate_arithmetic_kernel(seed: int | None = None) -> KernelSpec:
    """Elementwise arithmetic + unary/binary op chains."""
    if seed is not None:
        random.seed(seed)

    name = _random_name("arith")
    num_ops = random.randint(2, 10)

    ops: list[str] = []
    var_counter = 0

    for i in range(num_ops):
        if random.random() < 0.5:
            op_name = random.choice(list(UNARY_OPS.keys()))
            tpl = UNARY_OPS[op_name]
            expr = tpl.format(x="a[tid]" if i == 0 else f"var_{var_counter-1}")
        else:
            op_name = random.choice(list(BINARY_OPS.keys()))
            tpl = BINARY_OPS[op_name]
            if i == 0:
                expr = tpl.format(x="a[tid]", y="b[tid]")
            else:
                expr = tpl.format(x=f"var_{var_counter-1}", y="b[tid]")

        ops.append(f"    var_{var_counter} = {expr}")
        var_counter += 1

    source = f"""@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
{chr(10).join(ops)}
    out[tid] = var_{var_counter-1}
"""

    return KernelSpec(
        name=name,
        category="arithmetic",
        source=source,
        arg_types={
            "a": "wp.array(dtype=float)",
            "b": "wp.array(dtype=float)",
            "out": "wp.array(dtype=float)",
        },
        description=f"Arithmetic chain with {num_ops} ops",
        metadata={"num_ops": num_ops, "seed": seed},
    )


def generate_vector_kernel(seed: int | None = None) -> KernelSpec:
    """Vector ops on vec2/vec3/vec4."""
    if seed is not None:
        random.seed(seed)

    name = _random_name("vec")
    vec_type = random.choice(["wp.vec2", "wp.vec3", "wp.vec4"])
    op = random.choice(["dot", "length", "normalize"] + (["cross"] if vec_type == "wp.vec3" else []))

    if op == "dot":
        body = "    out[tid] = wp.dot(a[tid], b[tid])"
        out_type = "float"
    elif op == "cross":
        body = "    out[tid] = wp.cross(a[tid], b[tid])"
        out_type = vec_type
    elif op == "length":
        body = "    out[tid] = wp.length(a[tid])"
        out_type = "float"
    else:
        body = "    out[tid] = wp.normalize(a[tid])"
        out_type = vec_type

    source = f"""@wp.kernel
def {name}(a: wp.array(dtype={vec_type}), b: wp.array(dtype={vec_type}), out: wp.array(dtype={out_type})):
    tid = wp.tid()
{body}
"""

    return KernelSpec(
        name=name,
        category="vector",
        source=source,
        arg_types={
            "a": f"wp.array(dtype={vec_type})",
            "b": f"wp.array(dtype={vec_type})",
            "out": f"wp.array(dtype={out_type})",
        },
        description=f"Vector op {op} on {vec_type}",
        metadata={"vec_type": vec_type, "operation": op, "seed": seed},
    )


def generate_matrix_kernel(seed: int | None = None) -> KernelSpec:
    """Matrix ops on mat22/mat33/mat44."""
    if seed is not None:
        random.seed(seed)

    name = _random_name("mat")
    size = random.choice([2, 3, 4])
    mat_type = f"wp.mat{size}{size}"
    vec_type = f"wp.vec{size}"
    op = random.choice(["mat_vec", "mat_mat", "transpose"])

    if op == "mat_vec":
        source = f"""@wp.kernel
def {name}(m: wp.array(dtype={mat_type}), v: wp.array(dtype={vec_type}), out: wp.array(dtype={vec_type})):
    tid = wp.tid()
    out[tid] = m[tid] * v[tid]
"""
        arg_types = {"m": f"wp.array(dtype={mat_type})", "v": f"wp.array(dtype={vec_type})", "out": f"wp.array(dtype={vec_type})"}
        desc = f"Matrix-vector multiply ({mat_type} * {vec_type})"
    elif op == "mat_mat":
        source = f"""@wp.kernel
def {name}(a: wp.array(dtype={mat_type}), b: wp.array(dtype={mat_type}), out: wp.array(dtype={mat_type})):
    tid = wp.tid()
    out[tid] = a[tid] * b[tid]
"""
        arg_types = {"a": f"wp.array(dtype={mat_type})", "b": f"wp.array(dtype={mat_type})", "out": f"wp.array(dtype={mat_type})"}
        desc = f"Matrix-matrix multiply ({mat_type})"
    else:
        source = f"""@wp.kernel
def {name}(m: wp.array(dtype={mat_type}), out: wp.array(dtype={mat_type})):
    tid = wp.tid()
    out[tid] = wp.transpose(m[tid])
"""
        arg_types = {"m": f"wp.array(dtype={mat_type})", "out": f"wp.array(dtype={mat_type})"}
        desc = f"Matrix transpose ({mat_type})"

    return KernelSpec(
        name=name,
        category="matrix",
        source=source,
        arg_types=arg_types,
        description=desc,
        metadata={"mat_type": mat_type, "operation": op, "seed": seed},
    )


def generate_control_flow_kernel(seed: int | None = None) -> KernelSpec:
    """Control flow (if/elif/else, loops)."""
    if seed is not None:
        random.seed(seed)

    name = _random_name("ctrl")
    pattern = random.choice(["clamp", "abs_diff", "step", "loop_sum", "loop_product"])

    if pattern == "clamp":
        source = f"""@wp.kernel
def {name}(a: wp.array(dtype=float), lo: float, hi: float, out: wp.array(dtype=float)):
    tid = wp.tid()
    val = a[tid]
    if val < lo:
        out[tid] = lo
    elif val > hi:
        out[tid] = hi
    else:
        out[tid] = val
"""
        arg_types = {"a": "wp.array(dtype=float)", "lo": "float", "hi": "float", "out": "wp.array(dtype=float)"}
        desc = "Clamp values between lo and hi"
    elif pattern == "abs_diff":
        source = f"""@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    diff = a[tid] - b[tid]
    if diff < 0.0:
        out[tid] = -diff
    else:
        out[tid] = diff
"""
        arg_types = {"a": "wp.array(dtype=float)", "b": "wp.array(dtype=float)", "out": "wp.array(dtype=float)"}
        desc = "Absolute difference"
    elif pattern == "step":
        source = f"""@wp.kernel
def {name}(a: wp.array(dtype=float), threshold: float, out: wp.array(dtype=float)):
    tid = wp.tid()
    if a[tid] > threshold:
        out[tid] = 1.0
    else:
        out[tid] = 0.0
"""
        arg_types = {"a": "wp.array(dtype=float)", "threshold": "float", "out": "wp.array(dtype=float)"}
        desc = "Step function"
    elif pattern == "loop_sum":
        n_iters = random.randint(4, 16)
        source = f"""@wp.kernel
def {name}(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    total = float(0.0)
    for i in range({n_iters}):
        total = total + a[tid]
    out[tid] = total
"""
        arg_types = {"a": "wp.array(dtype=float)", "out": "wp.array(dtype=float)"}
        desc = f"Loop sum ({n_iters} iterations)"
    else:
        n_iters = random.randint(4, 12)
        source = f"""@wp.kernel
def {name}(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    prod = float(1.0)
    for i in range({n_iters}):
        prod = prod * (a[tid] + 0.0001)
    out[tid] = prod
"""
        arg_types = {"a": "wp.array(dtype=float)", "out": "wp.array(dtype=float)"}
        desc = f"Loop product ({n_iters} iterations)"

    return KernelSpec(
        name=name,
        category="control_flow",
        source=source,
        arg_types=arg_types,
        description=desc,
        metadata={"pattern": pattern, "seed": seed},
    )


def generate_math_kernel(seed: int | None = None) -> KernelSpec:
    """Math intrinsics (composition of unary ops)."""
    if seed is not None:
        random.seed(seed)

    name = _random_name("math")
    num_funcs = random.randint(2, 6)
    funcs = random.sample(list(UNARY_OPS.keys()), k=min(num_funcs, len(UNARY_OPS)))

    expr = "a[tid]"
    for func in funcs:
        expr = UNARY_OPS[func].format(x=expr)

    source = f"""@wp.kernel
def {name}(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = {expr}
"""

    return KernelSpec(
        name=name,
        category="math",
        source=source,
        arg_types={"a": "wp.array(dtype=float)", "out": "wp.array(dtype=float)"},
        description=f"Math chain: {', '.join(funcs)}",
        metadata={"functions": funcs, "seed": seed},
    )


def generate_atomic_kernel(seed: int | None = None) -> KernelSpec:
    """Atomic ops on a single-element result."""
    if seed is not None:
        random.seed(seed)

    name = _random_name("atom")
    op = random.choice(["add", "min", "max"])
    op_fn = {"add": "wp.atomic_add", "min": "wp.atomic_min", "max": "wp.atomic_max"}[op]

    source = f"""@wp.kernel
def {name}(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    {op_fn}(result, 0, values[tid])
"""

    return KernelSpec(
        name=name,
        category="atomic",
        source=source,
        arg_types={"values": "wp.array(dtype=float)", "result": "wp.array(dtype=float)"},
        description=f"Atomic {op} reduction",
        metadata={"operation": op, "seed": seed},
    )


def generate_nested_loop_kernel(seed: int | None = None) -> KernelSpec:
    """Nested loops (loop IR stress)."""
    if seed is not None:
        random.seed(seed)

    name = _random_name("nested")
    outer = random.randint(3, 8)
    inner = random.randint(3, 8)

    source = f"""@wp.kernel
def {name}(data: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    total = float(0.0)
    for i in range({outer}):
        for j in range({inner}):
            total = total + data[tid] * float(i * j + 1)
    out[tid] = total
"""

    return KernelSpec(
        name=name,
        category="nested_loop",
        source=source,
        arg_types={"data": "wp.array(dtype=float)", "out": "wp.array(dtype=float)"},
        description=f"Nested loops ({outer}x{inner})",
        metadata={"outer_iterations": outer, "inner_iterations": inner, "seed": seed},
    )


def generate_multi_conditional_kernel(seed: int | None = None) -> KernelSpec:
    """Multi-way branching."""
    if seed is not None:
        random.seed(seed)

    name = _random_name("multicond")
    t1 = round(random.uniform(-5, 5), 2)
    t2 = round(t1 + random.uniform(0.5, 5), 2)

    source = f"""@wp.kernel
def {name}(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    val = x[tid]
    if val < {t1}:
        out[tid] = val * 0.5
    elif val < {t2}:
        out[tid] = val * 1.0
    else:
        out[tid] = val * 2.0
"""

    return KernelSpec(
        name=name,
        category="multi_conditional",
        source=source,
        arg_types={"x": "wp.array(dtype=float)", "out": "wp.array(dtype=float)"},
        description=f"Multiple conditions (thresholds: {t1}, {t2})",
        metadata={"threshold1": t1, "threshold2": t2, "seed": seed},
    )


def generate_combined_kernel(seed: int | None = None) -> KernelSpec:
    """Combination of loops, conditionals, and math."""
    if seed is not None:
        random.seed(seed)

    name = _random_name("combined")
    iterations = random.randint(3, 10)
    threshold = round(random.uniform(-2, 2), 2)
    func = random.choice(["wp.sin", "wp.cos", "wp.sqrt", "wp.abs", "wp.exp"])

    source = f"""@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    acc = float(0.0)
    for i in range({iterations}):
        if a[tid] * float(i) > {threshold}:
            acc = acc + {func}(b[tid])
        else:
            acc = acc + b[tid]
    out[tid] = acc
"""

    return KernelSpec(
        name=name,
        category="combined",
        source=source,
        arg_types={"a": "wp.array(dtype=float)", "b": "wp.array(dtype=float)", "out": "wp.array(dtype=float)"},
        description=f"Combined pattern (loop+cond+{func})",
        metadata={"iterations": iterations, "threshold": threshold, "function": func, "seed": seed},
    )


def generate_scalar_param_kernel(seed: int | None = None) -> KernelSpec:
    """Scalar parameters (to stress arg lists)."""
    if seed is not None:
        random.seed(seed)

    name = _random_name("scalar")
    op = random.choice(["+", "-", "*"])
    scale = round(random.uniform(0.1, 10.0), 2)
    offset = round(random.uniform(-5, 5), 2)

    source = f"""@wp.kernel
def {name}(x: wp.array(dtype=float), out: wp.array(dtype=float), scale: float, offset: float):
    tid = wp.tid()
    out[tid] = (x[tid] {op} scale) + offset
"""

    return KernelSpec(
        name=name,
        category="scalar_param",
        source=source,
        arg_types={"x": "wp.array(dtype=float)", "out": "wp.array(dtype=float)", "scale": "float", "offset": "float"},
        description=f"Scalar parameters (scale={scale}, offset={offset})",
        metadata={"operation": op, "scale": scale, "offset": offset, "seed": seed},
    )


GENERATORS = {
    "arithmetic": generate_arithmetic_kernel,
    "vector": generate_vector_kernel,
    "matrix": generate_matrix_kernel,
    "control_flow": generate_control_flow_kernel,
    "math": generate_math_kernel,
    "atomic": generate_atomic_kernel,
    "nested_loop": generate_nested_loop_kernel,
    "multi_conditional": generate_multi_conditional_kernel,
    "combined": generate_combined_kernel,
    "scalar_param": generate_scalar_param_kernel,
}


def generate_kernel(category: str | None = None, seed: int | None = None) -> KernelSpec:
    if category is None:
        category = random.choice(list(GENERATORS.keys()))
    if category not in GENERATORS:
        raise ValueError(f"Unknown category: {category}. Available: {list(GENERATORS.keys())}")
    return GENERATORS[category](seed)


def generate_kernels(n: int, categories: list[str] | None = None, seed: int | None = None) -> list[KernelSpec]:
    if seed is not None:
        random.seed(seed)
    if categories is None:
        categories = list(GENERATORS.keys())

    out: list[KernelSpec] = []
    for i in range(n):
        cat = random.choice(categories)
        out.append(generate_kernel(cat, seed=(seed + i) if seed is not None else None))
    return out

