#!/usr/bin/env python3
"""
Test CUDA IR extraction for all kernel types.

These tests verify that CUDA IR can be extracted without requiring an actual GPU.
They test the code generation path, not execution.
"""
import sys
from pathlib import Path
import pytest

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code" / "synthesis"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code" / "extraction"))

import warp as wp
wp.init()

from generator import generate_kernel, GENERATORS
from pipeline import compile_kernel_from_source, extract_ir_from_kernel


class TestCUDAExtraction:
    """Test CUDA IR extraction for all kernel types."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        pass
    
    @pytest.mark.parametrize("kernel_type", list(GENERATORS.keys()))
    def test_forward_extraction(self, kernel_type):
        """Test that forward kernel can be extracted for CUDA."""
        spec = generate_kernel(kernel_type, seed=42)
        kernel = compile_kernel_from_source(spec.source, spec.name)
        
        ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=False)
        
        # Verify forward code was extracted
        assert ir["forward_code"] is not None, f"Forward code not extracted for {kernel_type}"
        
        # Verify CUDA-specific markers
        assert "_cuda_kernel_forward" in ir["forward_code"], \
            f"CUDA marker not found in forward code for {kernel_type}"
        
        # Verify CUDA threading model
        assert "blockDim" in ir["forward_code"] or "threadIdx" in ir["forward_code"], \
            f"CUDA threading not found in forward code for {kernel_type}"
    
    @pytest.mark.parametrize("kernel_type", list(GENERATORS.keys()))
    def test_backward_extraction(self, kernel_type):
        """Test that backward kernel can be extracted for CUDA."""
        spec = generate_kernel(kernel_type, seed=42)
        kernel = compile_kernel_from_source(spec.source, spec.name)
        
        ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=True)
        
        # Verify backward code was extracted
        assert ir["backward_code"] is not None, f"Backward code not extracted for {kernel_type}"
        
        # Verify CUDA-specific markers
        assert "_cuda_kernel_backward" in ir["backward_code"], \
            f"CUDA marker not found in backward code for {kernel_type}"
        
        # Verify adjoint variables present
        assert "adj_" in ir["backward_code"], \
            f"Adjoint variables not found in backward code for {kernel_type}"
    
    @pytest.mark.parametrize("kernel_type", list(GENERATORS.keys()))
    def test_cpu_vs_cuda_difference(self, kernel_type):
        """Test that CPU and CUDA IR are different."""
        spec = generate_kernel(kernel_type, seed=42)
        kernel = compile_kernel_from_source(spec.source, spec.name)
        
        cpu_ir = extract_ir_from_kernel(kernel, device="cpu", include_backward=False)
        cuda_ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=False)
        
        # Both should have forward code
        assert cpu_ir["forward_code"] is not None
        assert cuda_ir["forward_code"] is not None
        
        # They should be different
        assert cpu_ir["forward_code"] != cuda_ir["forward_code"], \
            f"CPU and CUDA IR should be different for {kernel_type}"
        
        # CPU should have _cpu_, CUDA should have _cuda_
        assert "_cpu_kernel_forward" in cpu_ir["forward_code"]
        assert "_cuda_kernel_forward" in cuda_ir["forward_code"]


class TestCUDAIRContent:
    """Test the content of generated CUDA IR."""
    
    def test_cuda_has_block_dim(self):
        """Verify CUDA IR uses CUDA block/thread model."""
        spec = generate_kernel("arithmetic", seed=100)
        kernel = compile_kernel_from_source(spec.source, spec.name)
        ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=False)
        
        forward = ir["forward_code"]
        assert "blockDim.x" in forward, "CUDA IR should use blockDim"
        assert "blockIdx.x" in forward, "CUDA IR should use blockIdx"
        assert "threadIdx.x" in forward, "CUDA IR should use threadIdx"
    
    def test_cuda_has_tile_shared_storage(self):
        """Verify CUDA IR has tile shared storage."""
        spec = generate_kernel("arithmetic", seed=100)
        kernel = compile_kernel_from_source(spec.source, spec.name)
        ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=False)
        
        forward = ir["forward_code"]
        assert "tile_shared_storage" in forward, "CUDA IR should have tile shared storage"
    
    def test_backward_has_adjoint_operations(self):
        """Verify backward kernel has adjoint operations."""
        spec = generate_kernel("arithmetic", seed=100)
        kernel = compile_kernel_from_source(spec.source, spec.name)
        ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=True)
        
        backward = ir["backward_code"]
        assert backward is not None, "Backward code should be generated"
        assert "adj_" in backward, "Backward should have adjoint variables"
        assert "// reverse" in backward or "adj_array" in backward, \
            "Backward should have reverse pass"


class TestAtomicKernels:
    """Test CUDA atomic operations."""
    
    def test_atomic_add_cuda(self):
        """Test CUDA atomic_add kernel extraction."""
        source = """@wp.kernel
def atomic_test(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(result, 0, values[tid])
"""
        kernel = compile_kernel_from_source(source, "atomic_test")
        ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=False)
        
        assert ir["forward_code"] is not None
        assert "_cuda_kernel_forward" in ir["forward_code"]
    
    def test_atomic_min_cuda(self):
        """Test CUDA atomic_min kernel extraction."""
        source = """@wp.kernel
def atomic_min_test(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_min(result, 0, values[tid])
"""
        kernel = compile_kernel_from_source(source, "atomic_min_test")
        ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=False)
        
        assert ir["forward_code"] is not None
        assert "_cuda_kernel_forward" in ir["forward_code"]


class TestVectorMatrixKernels:
    """Test CUDA vector and matrix operations."""
    
    def test_vector_dot_cuda(self):
        """Test CUDA vector dot product."""
        source = """@wp.kernel
def vec_dot_test(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.dot(a[tid], b[tid])
"""
        kernel = compile_kernel_from_source(source, "vec_dot_test")
        ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=True)
        
        assert ir["forward_code"] is not None
        assert ir["backward_code"] is not None
        assert "wp::dot" in ir["forward_code"]
    
    def test_matrix_multiply_cuda(self):
        """Test CUDA matrix-vector multiply."""
        source = """@wp.kernel
def mat_vec_test(m: wp.array(dtype=wp.mat33), v: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    out[tid] = m[tid] * v[tid]
"""
        kernel = compile_kernel_from_source(source, "mat_vec_test")
        ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=True)
        
        assert ir["forward_code"] is not None
        assert ir["backward_code"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
