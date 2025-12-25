"""Programmatic kernel generator for Python→IR training data synthesis."""
import random
import string
import os
import tempfile
import importlib.util


# Operation templates
UNARY_OPS = ['wp.sin', 'wp.cos', 'wp.exp', 'wp.sqrt', 'wp.abs', 'wp.log']
BINARY_OPS = ['+', '-', '*', '/']
COMPARE_OPS = ['>', '<', '>=', '<=', '==', '!=']
REDUCE_OPS = ['wp.min', 'wp.max']


def _random_name(prefix: str = "kernel", length: int = 6) -> str:
    """Generate a random kernel name."""
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}_{suffix}"


def generate_arithmetic_kernel(name: str = None, num_ops: int = 2) -> str:
    """Generate a kernel with arithmetic operations.
    
    Args:
        name: Kernel name (random if None)
        num_ops: Number of operations to chain
        
    Returns:
        Python source code for the kernel
    """
    name = name or _random_name("arith")
    
    # Build expression
    expr = "a[i]"
    for _ in range(num_ops):
        op = random.choice(BINARY_OPS)
        operand = random.choice(["b[i]", f"{random.uniform(0.1, 10.0):.2f}"])
        expr = f"({expr} {op} {operand})"
    
    return f'''@wp.kernel
def {name}(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    i = wp.tid()
    c[i] = {expr}
'''


def generate_math_kernel(name: str = None, num_funcs: int = 2) -> str:
    """Generate a kernel with math function calls.
    
    Args:
        name: Kernel name (random if None)
        num_funcs: Number of functions to chain
        
    Returns:
        Python source code for the kernel
    """
    name = name or _random_name("math")
    
    # Build nested function expression
    expr = "x[i]"
    for _ in range(num_funcs):
        func = random.choice(UNARY_OPS)
        expr = f"{func}({expr})"
    
    return f'''@wp.kernel
def {name}(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = {expr}
'''


def generate_loop_kernel(name: str = None, loop_bound: int = None) -> str:
    """Generate a kernel with a loop.
    
    Args:
        name: Kernel name (random if None)
        loop_bound: Static loop bound (random 3-10 if None)
        
    Returns:
        Python source code for the kernel
    """
    name = name or _random_name("loop")
    loop_bound = loop_bound or random.randint(3, 10)
    
    # Accumulation pattern
    op = random.choice(['+', '*'])
    init_val = "0.0" if op == '+' else "1.0"
    
    return f'''@wp.kernel
def {name}(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    acc = float({init_val})
    for j in range({loop_bound}):
        acc = acc {op} x[i]
    y[i] = acc
'''


def generate_conditional_kernel(name: str = None) -> str:
    """Generate a kernel with conditional logic.
    
    Args:
        name: Kernel name (random if None)
        
    Returns:
        Python source code for the kernel
    """
    name = name or _random_name("cond")
    
    cmp_op = random.choice(COMPARE_OPS)
    threshold = random.uniform(-1.0, 1.0)
    
    return f'''@wp.kernel
def {name}(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    val = x[i]
    if val {cmp_op} {threshold:.2f}:
        y[i] = val * 2.0
    else:
        y[i] = val * 0.5
'''


def generate_vector_kernel(name: str = None) -> str:
    """Generate a kernel with vector operations.
    
    Args:
        name: Kernel name (random if None)
        
    Returns:
        Python source code for the kernel
    """
    name = name or _random_name("vec")
    
    ops = ['wp.dot', 'wp.cross', 'wp.length', 'wp.normalize']
    op = random.choice(ops)
    
    if op == 'wp.dot':
        return f'''@wp.kernel
def {name}(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), c: wp.array(dtype=float)):
    i = wp.tid()
    c[i] = wp.dot(a[i], b[i])
'''
    elif op == 'wp.cross':
        return f'''@wp.kernel
def {name}(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), c: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    c[i] = wp.cross(a[i], b[i])
'''
    elif op == 'wp.length':
        return f'''@wp.kernel
def {name}(a: wp.array(dtype=wp.vec3), c: wp.array(dtype=float)):
    i = wp.tid()
    c[i] = wp.length(a[i])
'''
    else:  # normalize
        return f'''@wp.kernel
def {name}(a: wp.array(dtype=wp.vec3), c: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    c[i] = wp.normalize(a[i])
'''


def generate_matrix_kernel(name: str = None) -> str:
    """Generate a kernel with matrix operations.
    
    Args:
        name: Kernel name (random if None)
        
    Returns:
        Python source code for the kernel
    """
    name = name or _random_name("mat")
    
    # Matrix-vector multiply
    return f'''@wp.kernel
def {name}(A: wp.array(dtype=wp.mat33), v: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = A[i] @ v[i]
'''


def generate_combined_kernel(name: str = None) -> str:
    """Generate a kernel combining multiple patterns.
    
    Args:
        name: Kernel name (random if None)
        
    Returns:
        Python source code for the kernel
    """
    name = name or _random_name("combined")
    
    # Combine loop, conditional, and math
    func1 = random.choice(UNARY_OPS)
    func2 = random.choice(UNARY_OPS)
    cmp_op = random.choice(COMPARE_OPS)
    threshold = random.uniform(0.0, 1.0)
    loop_bound = random.randint(2, 5)
    
    return f'''@wp.kernel
def {name}(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    val = x[i]
    acc = float(0.0)
    for j in range({loop_bound}):
        if val {cmp_op} {threshold:.2f}:
            acc = acc + {func1}(val)
        else:
            acc = acc + {func2}(val)
    y[i] = acc
'''


# Generator registry
GENERATORS = {
    'arithmetic': generate_arithmetic_kernel,
    'math': generate_math_kernel,
    'loop': generate_loop_kernel,
    'conditional': generate_conditional_kernel,
    'vector': generate_vector_kernel,
    'matrix': generate_matrix_kernel,
    'combined': generate_combined_kernel,
}


def generate_random_kernel(kernel_type: str = None) -> tuple:
    """Generate a random kernel.
    
    Args:
        kernel_type: Specific type or None for random
        
    Returns:
        Tuple of (kernel_type, kernel_source)
    """
    if kernel_type is None:
        kernel_type = random.choice(list(GENERATORS.keys()))
    
    generator = GENERATORS[kernel_type]
    source = generator()
    
    return kernel_type, source


def compile_kernel_source(source: str, kernel_name: str = None):
    """Compile kernel source and return the kernel object.
    
    Args:
        source: Python kernel source code
        kernel_name: Name to extract (auto-detect if None)
        
    Returns:
        Compiled warp kernel object
    """
    import warp as wp
    
    # Auto-detect kernel name from source
    if kernel_name is None:
        for line in source.split('\n'):
            if line.strip().startswith('def '):
                kernel_name = line.split('(')[0].replace('def ', '').strip()
                break
    
    # Write to temp file (warp needs source file)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import warp as wp\n\n")
        f.write(source)
        temp_path = f.name
    
    try:
        # Import the module
        spec = importlib.util.spec_from_file_location("temp_kernel", temp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the kernel
        kernel = getattr(module, kernel_name)
        return kernel
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    print("Kernel Generator Demo")
    print("="*60)
    
    for kernel_type in GENERATORS.keys():
        print(f"\n{kernel_type.upper()} KERNEL:")
        print("-"*40)
        _, source = generate_random_kernel(kernel_type)
        print(source)
