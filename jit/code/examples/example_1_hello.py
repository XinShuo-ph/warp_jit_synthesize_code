import warp as wp
import numpy as np

wp.init()

@wp.kernel
def double_array(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] * 2.0

def run():
    n = 10
    a_np = np.ones(n, dtype=np.float32)
    a = wp.from_numpy(a_np, dtype=float)

    wp.launch(kernel=double_array, dim=n, inputs=[a])

    result_np = a.numpy()
    print(f"Input: {np.ones(n)}")
    print(f"Output: {result_np}")
    
    expected = np.ones(n) * 2.0
    assert np.allclose(result_np, expected)
    print("Test passed!")

if __name__ == "__main__":
    run()
