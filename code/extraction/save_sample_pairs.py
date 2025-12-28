"""Save sample Python→IR pairs to data directory."""
import sys
import os
import json
sys.path.insert(0, "/workspace/jit/code/extraction")

import warp as wp
import numpy as np
from ir_extractor import extract_ir

wp.init()

# Sample kernels
@wp.kernel
def kernel_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

@wp.kernel
def kernel_dot_product(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.dot(a[tid], b[tid])

@wp.kernel
def kernel_mat_mul(m: wp.array(dtype=wp.mat33), v: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    out[tid] = m[tid] * v[tid]

@wp.kernel
def kernel_clamp(arr: wp.array(dtype=float), min_val: float, max_val: float):
    tid = wp.tid()
    val = arr[tid]
    if val < min_val:
        arr[tid] = min_val
    elif val > max_val:
        arr[tid] = max_val

@wp.kernel
def kernel_sum_neighbors(input: wp.array(dtype=float), output: wp.array(dtype=float), width: int):
    tid = wp.tid()
    total = float(0.0)
    for i in range(-1, 2):
        idx = tid + i
        if idx >= 0 and idx < width:
            total = total + input[idx]
    output[tid] = total


def force_compile():
    """Force compilation of all kernels."""
    n = 10
    a = wp.array(np.ones(n, dtype=np.float32))
    b = wp.array(np.ones(n, dtype=np.float32))
    c = wp.zeros(n, dtype=float)
    wp.launch(kernel_add, dim=n, inputs=[a, b, c])
    
    v1 = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
    v2 = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
    out_f = wp.zeros(n, dtype=float)
    wp.launch(kernel_dot_product, dim=n, inputs=[v1, v2, out_f])
    
    mats = wp.array(np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(n, axis=0), dtype=wp.mat33)
    vecs = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
    out_v = wp.zeros(n, dtype=wp.vec3)
    wp.launch(kernel_mat_mul, dim=n, inputs=[mats, vecs, out_v])
    
    arr = wp.array(np.random.randn(n).astype(np.float32))
    wp.launch(kernel_clamp, dim=n, inputs=[arr, -0.5, 0.5])
    
    inp = wp.array(np.ones(n, dtype=np.float32))
    outp = wp.zeros(n, dtype=float)
    wp.launch(kernel_sum_neighbors, dim=n, inputs=[inp, outp, n])
    
    wp.synchronize()


def save_pairs(output_dir: str):
    """Save Python→IR pairs as JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    kernels = [
        kernel_add,
        kernel_dot_product,
        kernel_mat_mul,
        kernel_clamp,
        kernel_sum_neighbors,
    ]
    
    for kernel in kernels:
        result = extract_ir(kernel)
        
        pair = {
            "python_source": result["python_source"],
            "cpp_forward": result["forward_code"],
            "cpp_backward": result["backward_code"],
            "metadata": result["metadata"],
        }
        
        filename = f"{kernel.key}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(pair, f, indent=2)
        
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    force_compile()
    save_pairs("/workspace/jit/data/samples")
    print("\nDone! Saved 5 sample pairs.")
