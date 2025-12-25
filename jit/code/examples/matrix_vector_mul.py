import warp as wp
import numpy as np

@wp.kernel
def mat_vec_mul_kernel(matrix: wp.array(dtype=float, ndim=2), 
                       vector: wp.array(dtype=float), 
                       result: wp.array(dtype=float), 
                       rows: int, 
                       cols: int):
    tid = wp.tid()
    if tid < rows:
        sum = float(0.0)
        for j in range(cols):
            sum += matrix[tid, j] * vector[j]
        result[tid] = sum

def run_example():
    wp.init()
    rows = 10
    cols = 5
    
    # Create input
    matrix_np = np.random.rand(rows, cols).astype(np.float32)
    vector_np = np.random.rand(cols).astype(np.float32)
    
    matrix_wp = wp.from_numpy(matrix_np, dtype=float)
    vector_wp = wp.from_numpy(vector_np, dtype=float)
    result_wp = wp.zeros(rows, dtype=float)
    
    # Launch
    wp.launch(kernel=mat_vec_mul_kernel, dim=rows, inputs=[matrix_wp, vector_wp, result_wp, rows, cols])
    
    # Verify
    result_np = result_wp.numpy()
    expected = np.dot(matrix_np, vector_np)
    
    if np.allclose(result_np, expected, atol=1e-5):
        print("Matrix Vector Mul: SUCCESS")
        print(f"First 5 elements: {result_np[:5]}")
    else:
        print("Matrix Vector Mul: FAILED")
        print(f"Expected: {expected[:5]}")
        print(f"Got: {result_np[:5]}")

if __name__ == "__main__":
    run_example()
