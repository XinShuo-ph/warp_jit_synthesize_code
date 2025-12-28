#!/usr/bin/env python3
"""
Test CUDA kernel compilation and execution on GPU.

These tests require an actual GPU to run. They are skipped if no CUDA device is available.
Run these tests on a machine with a GPU to validate the full pipeline.
"""
import sys
from pathlib import Path
import pytest
import numpy as np

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code" / "synthesis"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code" / "extraction"))

import warp as wp
wp.init()

from generator import generate_kernel, GENERATORS
from pipeline import compile_kernel_from_source


# Skip all tests if CUDA is not available
CUDA_AVAILABLE = wp.is_cuda_available()
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


class TestCUDAKernelExecution:
    """Test kernel execution on GPU."""
    
    def test_simple_arithmetic_cuda(self):
        """Test simple arithmetic kernel on GPU."""
        source = """@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]
"""
        kernel = compile_kernel_from_source(source, "add_kernel")
        
        # Create test data
        n = 1024
        a_np = np.random.rand(n).astype(np.float32)
        b_np = np.random.rand(n).astype(np.float32)
        
        # Create warp arrays on GPU
        a = wp.array(a_np, dtype=float, device="cuda")
        b = wp.array(b_np, dtype=float, device="cuda")
        c = wp.zeros(n, dtype=float, device="cuda")
        
        # Launch kernel
        wp.launch(kernel, dim=n, inputs=[a, b, c], device="cuda")
        
        # Verify result
        c_np = c.numpy()
        expected = a_np + b_np
        np.testing.assert_allclose(c_np, expected, rtol=1e-5)
    
    def test_vector_operations_cuda(self):
        """Test vector operations on GPU."""
        source = """@wp.kernel
def vec_dot_kernel(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.dot(a[tid], b[tid])
"""
        kernel = compile_kernel_from_source(source, "vec_dot_kernel")
        
        n = 512
        a_np = np.random.rand(n, 3).astype(np.float32)
        b_np = np.random.rand(n, 3).astype(np.float32)
        
        a = wp.array(a_np, dtype=wp.vec3, device="cuda")
        b = wp.array(b_np, dtype=wp.vec3, device="cuda")
        out = wp.zeros(n, dtype=float, device="cuda")
        
        wp.launch(kernel, dim=n, inputs=[a, b, out], device="cuda")
        
        out_np = out.numpy()
        expected = np.sum(a_np * b_np, axis=1)
        np.testing.assert_allclose(out_np, expected, rtol=1e-5)
    
    def test_control_flow_cuda(self):
        """Test control flow on GPU."""
        source = """@wp.kernel
def clamp_kernel(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    val = x[tid]
    if val < 0.0:
        out[tid] = 0.0
    elif val > 1.0:
        out[tid] = 1.0
    else:
        out[tid] = val
"""
        kernel = compile_kernel_from_source(source, "clamp_kernel")
        
        n = 1024
        x_np = np.random.rand(n).astype(np.float32) * 2 - 0.5  # Range [-0.5, 1.5]
        
        x = wp.array(x_np, dtype=float, device="cuda")
        out = wp.zeros(n, dtype=float, device="cuda")
        
        wp.launch(kernel, dim=n, inputs=[x, out], device="cuda")
        
        out_np = out.numpy()
        expected = np.clip(x_np, 0.0, 1.0)
        np.testing.assert_allclose(out_np, expected, rtol=1e-5)
    
    def test_math_functions_cuda(self):
        """Test math functions on GPU."""
        source = """@wp.kernel
def math_kernel(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.sin(x[tid]) + wp.cos(x[tid])
"""
        kernel = compile_kernel_from_source(source, "math_kernel")
        
        n = 1024
        x_np = np.random.rand(n).astype(np.float32) * 2 * np.pi
        
        x = wp.array(x_np, dtype=float, device="cuda")
        out = wp.zeros(n, dtype=float, device="cuda")
        
        wp.launch(kernel, dim=n, inputs=[x, out], device="cuda")
        
        out_np = out.numpy()
        expected = np.sin(x_np) + np.cos(x_np)
        np.testing.assert_allclose(out_np, expected, rtol=1e-5)
    
    @pytest.mark.parametrize("kernel_type", ["arithmetic", "vector", "math", "control_flow"])
    def test_generated_kernel_execution(self, kernel_type):
        """Test that generated kernels can execute on GPU."""
        spec = generate_kernel(kernel_type, seed=42)
        kernel = compile_kernel_from_source(spec.source, spec.name)
        
        # Get kernel info
        arg_types = spec.arg_types
        n = 256
        
        # Create appropriate test arrays based on kernel type
        # This is a simplified test - just verify the kernel launches without error
        try:
            if kernel_type == "arithmetic":
                a = wp.array(np.random.rand(n).astype(np.float32), device="cuda")
                b = wp.array(np.random.rand(n).astype(np.float32), device="cuda")
                c = wp.zeros(n, dtype=float, device="cuda")
                wp.launch(kernel, dim=n, inputs=[a, b, c], device="cuda")
            elif kernel_type == "vector":
                # Vector kernels have varying signatures, just verify compilation
                pass
            elif kernel_type == "math":
                a = wp.array(np.random.rand(n).astype(np.float32), device="cuda")
                out = wp.zeros(n, dtype=float, device="cuda")
                wp.launch(kernel, dim=n, inputs=[a, out], device="cuda")
            elif kernel_type == "control_flow":
                a = wp.array(np.random.rand(n).astype(np.float32), device="cuda")
                out = wp.zeros(n, dtype=float, device="cuda")
                # Control flow kernels may have different args
                pass
            
            wp.synchronize()  # Wait for GPU completion
            
        except Exception as e:
            pytest.fail(f"Kernel {spec.name} failed to execute: {e}")


class TestCUDADeviceInfo:
    """Test CUDA device information."""
    
    def test_cuda_device_exists(self):
        """Verify CUDA device is available."""
        assert wp.is_cuda_available(), "CUDA should be available"
    
    def test_cuda_device_name(self):
        """Get CUDA device name."""
        devices = wp.get_cuda_devices()
        assert len(devices) > 0, "Should have at least one CUDA device"
        print(f"CUDA devices: {devices}")


class TestCUDABackwardPass:
    """Test backward pass execution on GPU."""
    
    def test_gradient_computation_cuda(self):
        """Test gradient computation on GPU."""
        source = """@wp.kernel
def square_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = x[tid] * x[tid]
"""
        kernel = compile_kernel_from_source(source, "square_kernel")
        
        n = 256
        x_np = np.random.rand(n).astype(np.float32)
        
        x = wp.array(x_np, dtype=float, device="cuda", requires_grad=True)
        y = wp.zeros(n, dtype=float, device="cuda", requires_grad=True)
        
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=n, inputs=[x, y], device="cuda")
        
        # Set output gradient to 1.0
        y.grad = wp.array(np.ones(n, dtype=np.float32), device="cuda")
        
        # Compute backward pass
        tape.backward()
        
        # Gradient of x^2 is 2*x
        x_grad = x.grad.numpy()
        expected_grad = 2.0 * x_np
        np.testing.assert_allclose(x_grad, expected_grad, rtol=1e-5)


if __name__ == "__main__":
    if CUDA_AVAILABLE:
        print("CUDA is available, running GPU tests...")
        pytest.main([__file__, "-v"])
    else:
        print("CUDA is not available. These tests require a GPU.")
        print("Run on a machine with a GPU to validate kernel execution.")
