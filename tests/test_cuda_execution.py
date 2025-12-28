import unittest
import warp as wp
import numpy as np

@wp.kernel
def simple_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

class TestCUDAExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
    
    def test_cuda_launch(self):
        if not wp.is_cuda_available():
            print("Skipping CUDA test: GPU not available")
            return

        n = 100
        a = wp.array(np.ones(n, dtype=np.float32), device="cuda")
        b = wp.array(np.ones(n, dtype=np.float32) * 2.0, device="cuda")
        c = wp.zeros(n, dtype=float, device="cuda")
        
        wp.launch(kernel=simple_add, dim=n, inputs=[a, b, c], device="cuda")
        
        # Verify
        expected = np.ones(n) * 3.0
        actual = c.numpy()
        np.testing.assert_allclose(actual, expected)
        print("CUDA execution successful")

    def test_device_query(self):
        if not wp.is_cuda_available():
            print("Skipping CUDA test: GPU not available")
            return
            
        devices = wp.get_cuda_devices()
        self.assertTrue(len(devices) > 0)
        print(f"Found {len(devices)} CUDA devices")

if __name__ == '__main__':
    unittest.main()
