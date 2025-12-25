import random
import ast
import warp as wp

class KernelGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        
        self.var_names = ['a', 'b', 'c', 'x', 'y', 'z', 'val', 'temp']
        self.ops = ['+', '-', '*', '/']
        self.dtypes = ['float', 'int']
        self.math_funcs = ['wp.sin', 'wp.cos', 'wp.exp', 'wp.abs', 'wp.sqrt']

    def generate_kernel_code(self, kernel_name="generated_kernel"):
        """
        Generates a random Warp kernel as a string.
        """
        strategy = random.choice([
            'elementwise', 
            'conditional', 
            'loop',
            'vec3_op',
            'atomic_accumulate',
            'nested_loop',
            'complex_math'
        ])
        
        if strategy == 'elementwise':
            return self._generate_elementwise(kernel_name)
        elif strategy == 'conditional':
            return self._generate_conditional(kernel_name)
        elif strategy == 'loop':
            return self._generate_loop(kernel_name)
        elif strategy == 'vec3_op':
            return self._generate_vec3_op(kernel_name)
        elif strategy == 'atomic_accumulate':
            return self._generate_atomic_accumulate(kernel_name)
        elif strategy == 'nested_loop':
            return self._generate_nested_loop(kernel_name)
        elif strategy == 'complex_math':
            return self._generate_complex_math(kernel_name)
            
    def _generate_elementwise(self, name):
        # inputs: 2 arrays, output: 1 array
        dtype = random.choice(self.dtypes)
        code = f"""
@wp.kernel
def {name}(in1: wp.array(dtype={dtype}), in2: wp.array(dtype={dtype}), out: wp.array(dtype={dtype})):
    tid = wp.tid()
    x = in1[tid]
    y = in2[tid]
    out[tid] = x {random.choice(self.ops)} y
"""
        return code

    def _generate_conditional(self, name):
        dtype = 'float'
        op = random.choice(['>', '<', '>=', '<='])
        thresh = round(random.uniform(0, 10), 2)
        code = f"""
@wp.kernel
def {name}(data: wp.array(dtype={dtype}), out: wp.array(dtype={dtype})):
    tid = wp.tid()
    val = data[tid]
    if val {op} {thresh}:
        out[tid] = val * 2.0
    else:
        out[tid] = val
"""
        return code

    def _generate_loop(self, name):
        # Simple local loop
        code = f"""
@wp.kernel
def {name}(n: int, out: wp.array(dtype=float)):
    tid = wp.tid()
    sum = float(0.0)
    for i in range(n):
        sum = sum + float(i) * 0.1
    out[tid] = sum
"""
        return code

    def _generate_vec3_op(self, name):
        op = random.choice(['+', '-', '*'])
        if op == '*':
            # scalar mul
            op_str = "* 2.0"
        else:
            op_str = f"{op} b[tid]"
            
        code = f"""
@wp.kernel
def {name}(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    v = a[tid]
    out[tid] = v {op_str}
"""
        return code

    def _generate_atomic_accumulate(self, name):
        code = f"""
@wp.kernel
def {name}(data: wp.array(dtype=int), counter: wp.array(dtype=int)):
    tid = wp.tid()
    val = data[tid]
    if val > 0:
        wp.atomic_add(counter, 0, val)
"""
        return code

    def _generate_nested_loop(self, name):
        code = f"""
@wp.kernel
def {name}(width: int, height: int, out: wp.array(dtype=float)):
    tid = wp.tid()
    sum = float(0.0)
    for i in range(width):
        for j in range(height):
            sum = sum + float(i) * float(j)
    out[tid] = sum
"""
        return code

    def _generate_complex_math(self, name):
        func1 = random.choice(self.math_funcs)
        func2 = random.choice(self.math_funcs)
        code = f"""
@wp.kernel
def {name}(inp: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    x = inp[tid]
    # Ensure safe args for sqrt/log if needed (using abs)
    v1 = {func1}(x)
    v2 = {func2}(x)
    out[tid] = v1 + v2
"""
        return code

if __name__ == "__main__":
    gen = KernelGenerator()
    print(gen.generate_kernel_code("test_k"))
