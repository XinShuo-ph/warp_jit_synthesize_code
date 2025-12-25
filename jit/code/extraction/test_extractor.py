import unittest
import warp as wp
import numpy as np
import sys
import os

# Ensure we can import the extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from jit.code.extraction.ir_extractor import extract_ir

class TestIRExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_simple_math(self):
        @wp.kernel
        def math_kernel(a: float, b: float, out: wp.array(dtype=float)):
            tid = wp.tid()
            out[tid] = a * b + 2.0

        data = extract_ir(math_kernel)
        self.assertIn("ir", data)
        ir_text = "\n".join(data["ir"])
        self.assertIn("wp::mul", ir_text)
        self.assertIn("wp::add", ir_text)
        self.assertIn("builtin_tid1d", ir_text)

    def test_control_flow(self):
        @wp.kernel
        def flow_kernel(x: wp.array(dtype=float)):
            tid = wp.tid()
            v = x[tid]
            if v > 0.0:
                x[tid] = v
            else:
                x[tid] = 0.0

        data = extract_ir(flow_kernel)
        ir_text = "\n".join(data["ir"])
        # In SSA form, if/else often becomes phi nodes or branches
        # Warp IR usually keeps if/else structure or unrolls
        # Let's check for "if" or keywords. 
        # Actually warp codegen usually emits `if (...) {` structure in C++.
        # Wait, the IR I saw earlier was just statements.
        # Let's see how if is represented in the IR blocks.
        # It typically has multiple blocks and jump statements.
        # The extractor concatenates blocks.
        
        # Check if we have multiple blocks commented
        self.assertTrue(any("// Block" in line for line in data["ir"]))

    def test_loop(self):
        @wp.kernel
        def loop_kernel(arr: wp.array(dtype=float), n: int):
            for i in range(n):
                arr[i] = float(i)

        data = extract_ir(loop_kernel)
        ir_text = "\n".join(data["ir"])
        # Check for loop constructs or block structure
        self.assertTrue(len(data["ir"]) > 5)

    def test_struct(self):
        @wp.struct
        class MyStruct:
            val: float

        @wp.kernel
        def struct_kernel(s: MyStruct):
            s.val = 1.0

        # Note: Structs as kernel args usually need to be arrays of structs or passed in specific way.
        # But for extraction we just need compilation to succeed.
        
        data = extract_ir(struct_kernel)
        self.assertIn("ir", data)
        # Check if struct member access is in IR
        # Probably something like var_s.val or address calculation
        ir_text = "\n".join(data["ir"])
        # Just checking it doesn't crash and returns IR
        self.assertTrue(len(data["ir"]) > 0)

    def test_func_call(self):
        @wp.func
        def square(x: float):
            return x * x

        @wp.kernel
        def call_kernel(out: wp.array(dtype=float)):
            tid = wp.tid()
            out[tid] = square(2.0)

        data = extract_ir(call_kernel)
        ir_text = "\n".join(data["ir"])
        # Function calls might be inlined or called.
        # Warp typically inlines user functions in kernels.
        # Update: It seems it generates a call in the IR block.
        self.assertTrue("square" in ir_text or "wp::mul" in ir_text)

if __name__ == "__main__":
    unittest.main()
