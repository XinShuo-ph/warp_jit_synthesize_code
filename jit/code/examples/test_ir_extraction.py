import warp as wp
import sys
import os

# Add workspace root to path so we can import jit package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from jit.code.extraction.ir_extractor import get_kernel_ir

wp.init()

# 1. Simple Math
@wp.kernel
def math_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

# 2. Loops
@wp.kernel
def loop_kernel(n: int, result: wp.array(dtype=float)):
    tid = wp.tid()
    sum = float(0.0)
    for i in range(n):
        sum = sum + float(i)
    result[tid] = sum

# 3. Structs
@wp.struct
class MyStruct:
    x: float
    y: float

@wp.kernel
def struct_kernel(input: wp.array(dtype=MyStruct), output: wp.array(dtype=float)):
    tid = wp.tid()
    s = input[tid]
    output[tid] = s.x + s.y

# 4. Atomic
@wp.kernel
def atomic_kernel(counter: wp.array(dtype=int)):
    tid = wp.tid()
    wp.atomic_add(counter, 0, 1)

# 5. Conditional
@wp.kernel
def conditional_kernel(data: wp.array(dtype=float), threshold: float):
    tid = wp.tid()
    val = data[tid]
    if val > threshold:
        data[tid] = val * 2.0
    else:
        data[tid] = 0.0

kernels = [
    ("math_kernel", math_kernel),
    ("loop_kernel", loop_kernel),
    ("struct_kernel", struct_kernel),
    ("atomic_kernel", atomic_kernel),
    ("conditional_kernel", conditional_kernel)
]

for name, k in kernels:
    print(f"--- Extracting {name} ---")
    try:
        ir = get_kernel_ir(k, device="cuda")
        print(f"IR length: {len(ir)}")
        print("Start of IR:")
        print(ir[:200] + "...")
        print("Success!\n")
    except Exception as e:
        print(f"Failed to extract {name}: {e}\n")
