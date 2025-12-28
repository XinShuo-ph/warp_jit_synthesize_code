"""
Example 2: Vector operations with warp math functions
"""
import warp as wp
import numpy as np

wp.init()

@wp.kernel
def vector_length(positions: wp.array(dtype=wp.vec3),
                   lengths: wp.array(dtype=float)):
    """Calculate length of 3D vectors"""
    tid = wp.tid()
    pos = positions[tid]
    lengths[tid] = wp.length(pos)

@wp.kernel
def normalize_vectors(input_vecs: wp.array(dtype=wp.vec3),
                       output_vecs: wp.array(dtype=wp.vec3)):
    """Normalize 3D vectors"""
    tid = wp.tid()
    v = input_vecs[tid]
    output_vecs[tid] = wp.normalize(v)

def run_example():
    # Create test vectors
    n = 5
    positions_np = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [3.0, 4.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0]
    ], dtype=np.float32)
    
    # Create warp arrays
    positions = wp.array(positions_np, dtype=wp.vec3)
    lengths = wp.zeros(n, dtype=wp.float32)
    normalized = wp.zeros(n, dtype=wp.vec3)
    
    # Launch kernels
    wp.launch(vector_length, dim=n, inputs=[positions, lengths])
    wp.launch(normalize_vectors, dim=n, inputs=[positions, normalized])
    
    # Get results
    lengths_result = lengths.numpy()
    normalized_result = normalized.numpy()
    
    print("Example 2: Vector Operations")
    print(f"Input positions: ")
    for i, p in enumerate(positions_np):
        print(f"  [{i}]: {p}")
    print(f"\nVector lengths: {lengths_result}")
    print(f"\nNormalized vectors:")
    for i, v in enumerate(normalized_result):
        print(f"  [{i}]: {v}, length: {np.linalg.norm(v):.6f}")
    
    # Verify normalized vectors have unit length
    normalized_lengths = np.array([np.linalg.norm(v) for v in normalized_result])
    success = np.allclose(normalized_lengths, 1.0, rtol=1e-5)
    print(f"\nAll normalized vectors have unit length: {success}")
    
    return success

if __name__ == "__main__":
    success = run_example()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
