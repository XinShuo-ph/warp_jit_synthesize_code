import warp as wp
import numpy as np
import sys

def smoke_test():
    print("Initializing Warp...")
    wp.init()
    
    if not wp.is_cuda_available():
        print("❌ Error: CUDA is not available in this environment.")
        sys.exit(1)
    
    device = "cuda:0"
    print(f"✅ CUDA available. Using device: {device}")
    
    n = 1024
    a = wp.array(np.ones(n, dtype=np.float32), device=device)
    b = wp.array(np.full(n, 2.0, dtype=np.float32), device=device)
    c = wp.zeros(n, dtype=np.float32, device=device)
    
    @wp.kernel
    def add_arrays(x: wp.array(dtype=float), y: wp.array(dtype=float), z: wp.array(dtype=float)):
        tid = wp.tid()
        z[tid] = x[tid] + y[tid]
    
    print("Launching kernel...")
    wp.launch(kernel=add_arrays, dim=n, inputs=[a, b, c], device=device)
    
    # Synchronize
    wp.synchronize()
    
    # Verify
    result = c.numpy()
    expected = np.ones(n) * 3.0
    
    if np.allclose(result, expected):
        print("✅ Smoke test passed! Kernel execution successful.")
    else:
        print("❌ Smoke test failed! Results do not match.")
        print(f"Expected: {expected[:5]}")
        print(f"Got: {result[:5]}")
        sys.exit(1)

if __name__ == "__main__":
    smoke_test()
