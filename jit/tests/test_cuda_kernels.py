"""CUDA Kernel Tests - Run on GPU hardware to validate CUDA IR generation.

This test suite validates:
1. All kernel types compile and run on CUDA
2. CUDA IR extraction works correctly
3. Forward and backward passes execute without errors

Requirements:
- NVIDIA GPU with CUDA driver
- warp-lang package installed

Usage:
    python -m pytest tests/test_cuda_kernels.py -v
"""
import os
import sys
import pytest
from pathlib import Path

# Add code paths
sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "extraction"))
sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "synthesis"))

import warp as wp


# Check CUDA availability
def cuda_available():
    try:
        wp.init()
        devices = wp.get_devices()
        return any("cuda" in str(d) for d in devices)
    except Exception:
        return False


CUDA_AVAILABLE = cuda_available()
SKIP_REASON = "CUDA not available - run on GPU hardware"


@pytest.fixture(scope="module")
def init_warp():
    """Initialize warp once per test module."""
    wp.init()
    yield


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
class TestCUDAExtraction:
    """Test CUDA IR extraction for all kernel types."""
    
    def test_arithmetic_kernel_cuda(self, init_warp):
        """Test arithmetic kernel on CUDA."""
        from ir_extractor import extract_ir, extract_kernel_functions
        
        @wp.kernel
        def arithmetic_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
            tid = wp.tid()
            c[tid] = a[tid] + b[tid] * 2.0
        
        ir_pair = extract_ir(arithmetic_add, device="cuda")
        
        assert ir_pair.device == "cuda"
        assert "arithmetic_add" in ir_pair.python_source
        assert len(ir_pair.cpp_ir) > 0
        
        # Use actual kernel key and check for CUDA-specific patterns
        funcs = extract_kernel_functions(ir_pair.cpp_ir, arithmetic_add.key, device="cuda")
        assert "forward" in funcs
        assert "__global__" in funcs["forward"] or "cuda_kernel_forward" in funcs["forward"]
    
    def test_math_kernel_cuda(self, init_warp):
        """Test math function kernel on CUDA."""
        from ir_extractor import extract_ir, extract_kernel_functions
        
        @wp.kernel
        def math_ops(x: wp.array(dtype=float), out: wp.array(dtype=float)):
            tid = wp.tid()
            out[tid] = wp.sin(x[tid]) + wp.cos(x[tid])
        
        ir_pair = extract_ir(math_ops, device="cuda")
        
        assert ir_pair.device == "cuda"
        funcs = extract_kernel_functions(ir_pair.cpp_ir, math_ops.key, device="cuda")
        assert len(funcs.get("forward", "")) > 0
    
    def test_conditional_kernel_cuda(self, init_warp):
        """Test conditional kernel on CUDA."""
        from ir_extractor import extract_ir, extract_kernel_functions
        
        @wp.kernel
        def conditional_test(x: wp.array(dtype=float), out: wp.array(dtype=float)):
            tid = wp.tid()
            if x[tid] > 0.5:
                out[tid] = x[tid] * 2.0
            else:
                out[tid] = x[tid] * 0.5
        
        ir_pair = extract_ir(conditional_test, device="cuda")
        
        assert ir_pair.device == "cuda"
        funcs = extract_kernel_functions(ir_pair.cpp_ir, conditional_test.key, device="cuda")
        assert len(funcs.get("forward", "")) > 0
    
    def test_loop_kernel_cuda(self, init_warp):
        """Test loop kernel on CUDA."""
        from ir_extractor import extract_ir, extract_kernel_functions
        
        @wp.kernel
        def loop_test(arr: wp.array(dtype=float), out: wp.array(dtype=float)):
            tid = wp.tid()
            acc = float(0.0)
            for i in range(4):
                acc = acc + arr[tid] * float(i)
            out[tid] = acc
        
        ir_pair = extract_ir(loop_test, device="cuda")
        
        assert ir_pair.device == "cuda"
        funcs = extract_kernel_functions(ir_pair.cpp_ir, loop_test.key, device="cuda")
        assert len(funcs.get("forward", "")) > 0
    
    def test_vector_kernel_cuda(self, init_warp):
        """Test vector operations kernel on CUDA."""
        from ir_extractor import extract_ir, extract_kernel_functions
        
        @wp.kernel
        def vector_test(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3)):
            tid = wp.tid()
            dt = 0.1
            pos[tid] = pos[tid] + vel[tid] * dt
        
        ir_pair = extract_ir(vector_test, device="cuda")
        
        assert ir_pair.device == "cuda"
        funcs = extract_kernel_functions(ir_pair.cpp_ir, vector_test.key, device="cuda")
        assert len(funcs.get("forward", "")) > 0
    
    def test_atomic_kernel_cuda(self, init_warp):
        """Test atomic operations kernel on CUDA."""
        from ir_extractor import extract_ir, extract_kernel_functions
        
        @wp.kernel
        def atomic_test(values: wp.array(dtype=float), result: wp.array(dtype=float)):
            tid = wp.tid()
            wp.atomic_add(result, 0, values[tid])
        
        ir_pair = extract_ir(atomic_test, device="cuda")
        
        assert ir_pair.device == "cuda"
        funcs = extract_kernel_functions(ir_pair.cpp_ir, atomic_test.key, device="cuda")
        assert len(funcs.get("forward", "")) > 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
class TestCUDAPipeline:
    """Test CUDA synthesis pipeline."""
    
    def test_pipeline_single_pair(self, init_warp, tmp_path):
        """Test generating a single CUDA pair."""
        from pipeline import SynthesisPipeline
        
        pipeline = SynthesisPipeline(str(tmp_path), seed=42, device="cuda")
        pair = pipeline.generate_pair("arithmetic")
        
        assert pair is not None
        assert pair.device == "cuda"
        assert "cuda_kernel_forward" in pair.cpp_ir_forward or "__global__" in pair.cpp_ir_forward
    
    def test_pipeline_batch(self, init_warp, tmp_path):
        """Test generating a batch of CUDA pairs."""
        from pipeline import SynthesisPipeline
        
        pipeline = SynthesisPipeline(str(tmp_path), seed=42, device="cuda")
        pairs = pipeline.generate_batch(5, kernel_types=["arithmetic", "conditional", "loop", "math", "vector"])
        
        assert len(pairs) == 5
        for pair in pairs:
            assert pair.device == "cuda"
        
        # Check files were saved with device field
        import json
        for json_file in tmp_path.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
            assert data["device"] == "cuda"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
class TestCUDAExecution:
    """Test actual CUDA kernel execution."""
    
    def test_kernel_execution(self, init_warp):
        """Test that kernels actually execute on CUDA."""
        import numpy as np
        
        @wp.kernel
        def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
            tid = wp.tid()
            c[tid] = a[tid] + b[tid]
        
        n = 1024
        a = wp.array(np.ones(n, dtype=np.float32), device="cuda")
        b = wp.array(np.ones(n, dtype=np.float32) * 2.0, device="cuda")
        c = wp.zeros(n, dtype=float, device="cuda")
        
        wp.launch(add_kernel, dim=n, inputs=[a, b, c], device="cuda")
        wp.synchronize_device("cuda")
        
        result = c.numpy()
        expected = np.ones(n) * 3.0
        np.testing.assert_array_almost_equal(result, expected)


# CPU fallback tests (always run)
class TestCPUExtraction:
    """CPU extraction tests - always run for regression testing."""
    
    def test_cpu_extraction(self, init_warp):
        """Test CPU extraction works."""
        from ir_extractor import extract_ir, extract_kernel_functions
        
        @wp.kernel
        def simple_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
            tid = wp.tid()
            b[tid] = a[tid] * 2.0
        
        ir_pair = extract_ir(simple_kernel, device="cpu")
        
        assert ir_pair.device == "cpu"
        assert "simple_kernel" in ir_pair.python_source
        
        # Use the actual kernel key which includes full scope path
        kernel_key = simple_kernel.key
        funcs = extract_kernel_functions(ir_pair.cpp_ir, kernel_key, device="cpu")
        assert "forward" in funcs
        assert "cpu_kernel_forward" in funcs["forward"]


if __name__ == "__main__":
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print("Running CUDA tests...")
    else:
        print("CUDA not available - only CPU tests will run")
        print("To run CUDA tests, execute on a machine with GPU hardware")
    
    pytest.main([__file__, "-v"])
