import warp as wp
import unittest
import sys
import os

# Add workspace root to path
sys.path.append(os.getcwd())

from jit.code.extraction.ir_extractor import get_kernel_ir

wp.init()

class TestIRExtractor(unittest.TestCase):
    
    def test_arithmetic_kernel(self):
        @wp.kernel
        def arithmetic(a: float, b: float, res: wp.array(dtype=float)):
            res[0] = a + b * 2.0 - 1.0

        ir = get_kernel_ir(arithmetic, device="cpu")
        self.assertIn("forward", ir)
        code = ir["forward"]
        
        # Check for C++ arithmetic ops (Warp uses functions like wp::add, wp::mul)
        self.assertIn("wp::mul", code)
        self.assertIn("wp::add", code)
        self.assertIn("wp::sub", code)
        self.assertIn("wp::array_t<wp::float32>", code)
        
    def test_loop_kernel(self):
        @wp.kernel
        def loop_k(n: int, res: wp.array(dtype=int)):
            sum = int(0) # Use int() to declare dynamic variable
            for i in range(n):
                sum = sum + i
            res[0] = sum

        ir = get_kernel_ir(loop_k, device="cpu")
        code = ir["forward"]
        
        self.assertTrue("for (" in code or "start_for" in code)
        self.assertTrue("++" in code or "iter_next" in code) # Increment
        
    def test_conditional_kernel(self):
        @wp.kernel
        def cond_k(x: float, res: wp.array(dtype=float)):
            if x > 0.0:
                res[0] = x
            else:
                res[0] = -x

        ir = get_kernel_ir(cond_k, device="cpu")
        code = ir["forward"]
        
        self.assertIn("if (", code)
        # Warp generates if (cond) ... if (!cond) instead of else sometimes
        self.assertTrue("else" in code or "if (!" in code)
        
    def test_array_access(self):
        @wp.kernel
        def array_k(arr: wp.array(dtype=float), idx: int):
            v = arr[idx]
            arr[idx] = v + 1.0

        ir = get_kernel_ir(array_k, device="cpu")
        code = ir["forward"]
        
        # Warp uses wp::load / wp::array_store or wp::address
        self.assertTrue("wp::load" in code or "wp::address" in code)
        self.assertIn("wp::array_store", code)
        
    def test_builtin_kernel(self):
        @wp.kernel
        def builtin_k(x: float, res: wp.array(dtype=float)):
            res[0] = wp.sin(x) * wp.exp(x)

        ir = get_kernel_ir(builtin_k, device="cpu")
        code = ir["forward"]
        
        self.assertIn("wp::sin(", code)
        self.assertIn("wp::exp(", code)

if __name__ == "__main__":
    unittest.main()
