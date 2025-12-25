import warp as wp
import numpy as np

@wp.kernel
def sine_kernel(input: wp.array(dtype=float), output: wp.array(dtype=float), freq: float, n: int):
    tid = wp.tid()
    if tid < n:
        output[tid] = wp.sin(input[tid] * freq)

def run_example():
    wp.init()
    n = 100
    freq = 2.0
    
    # Create input
    input_np = np.linspace(0, np.pi, n).astype(np.float32)
    input_wp = wp.from_numpy(input_np, dtype=float)
    output_wp = wp.zeros(n, dtype=float)
    
    # Launch
    wp.launch(kernel=sine_kernel, dim=n, inputs=[input_wp, output_wp, freq, n])
    
    # Verify
    output_np = output_wp.numpy()
    expected = np.sin(input_np * freq)
    
    if np.allclose(output_np, expected, atol=1e-5):
        print("Sine Wave: SUCCESS")
        print(f"First 5 elements: {output_np[:5]}")
    else:
        print("Sine Wave: FAILED")
        print(f"Expected: {expected[:5]}")
        print(f"Got: {output_np[:5]}")

if __name__ == "__main__":
    run_example()
