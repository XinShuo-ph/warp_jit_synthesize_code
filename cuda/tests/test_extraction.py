"""
Test CUDA IR extraction - runs without GPU.

These tests verify that CUDA code is generated correctly from Python kernels.
No actual GPU execution is performed.
"""
import sys
from pathlib import Path

# Add code directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "synthesis"))
sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "extraction"))

import warp as wp

# Initialize warp (CPU-only is fine)
wp.init()

from generator import KernelGenerator
from pipeline import compile_kernel_from_source, extract_ir_from_kernel, synthesize_pair


def test_cuda_forward_extraction():
    """Test that CUDA forward code is extracted correctly."""
    gen = KernelGenerator(seed=42)
    spec = gen.generate("arithmetic")
    source = gen.to_python_source(spec)
    
    kernel = compile_kernel_from_source(source, spec.name)
    ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=False)
    
    assert ir["forward_code"] is not None, "Forward code should be extracted"
    assert "cuda_kernel_forward" in ir["forward_code"], "Should be CUDA forward kernel"
    assert "blockDim" in ir["forward_code"], "Should contain CUDA thread indexing"
    assert "threadIdx" in ir["forward_code"], "Should contain CUDA thread indexing"
    
    print("✓ test_cuda_forward_extraction passed")


def test_cuda_backward_extraction():
    """Test that CUDA backward code is extracted correctly."""
    gen = KernelGenerator(seed=42)
    spec = gen.generate("arithmetic")
    source = gen.to_python_source(spec)
    
    kernel = compile_kernel_from_source(source, spec.name)
    ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=True)
    
    assert ir["forward_code"] is not None, "Forward code should be extracted"
    assert ir["backward_code"] is not None, "Backward code should be extracted"
    assert "cuda_kernel_backward" in ir["backward_code"], "Should be CUDA backward kernel"
    assert "adj_" in ir["backward_code"], "Backward should contain adjoint variables"
    
    print("✓ test_cuda_backward_extraction passed")


def test_cpu_vs_cuda_difference():
    """Verify CPU and CUDA code are different."""
    gen = KernelGenerator(seed=42)
    spec = gen.generate("arithmetic")
    source = gen.to_python_source(spec)
    
    kernel = compile_kernel_from_source(source, spec.name)
    
    cpu_ir = extract_ir_from_kernel(kernel, device="cpu", include_backward=False)
    cuda_ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=False)
    
    assert cpu_ir["forward_code"] is not None
    assert cuda_ir["forward_code"] is not None
    assert cpu_ir["forward_code"] != cuda_ir["forward_code"], "CPU and CUDA code should differ"
    
    assert "cpu_kernel_forward" in cpu_ir["forward_code"]
    assert "cuda_kernel_forward" in cuda_ir["forward_code"]
    
    print("✓ test_cpu_vs_cuda_difference passed")


def test_synthesize_pair_cuda():
    """Test full synthesis pipeline for CUDA."""
    gen = KernelGenerator(seed=42)
    spec = gen.generate("vector")
    source = gen.to_python_source(spec)
    
    pair = synthesize_pair(source, spec.name, "vector", device="cuda", include_backward=True)
    
    assert pair is not None, "Pair should be synthesized"
    assert pair["device"] == "cuda"
    assert "cuda_forward" in pair
    assert "cuda_backward" in pair
    assert pair["kernel_name"] == spec.name
    assert pair["python_source"] == source
    
    print("✓ test_synthesize_pair_cuda passed")


def test_all_kernel_types():
    """Test all kernel types generate CUDA code."""
    kernel_types = [
        'arithmetic', 'conditional', 'loop', 'math', 'vector', 
        'atomic', 'nested', 'multi_cond', 'combined', 'scalar_param', 'random_math'
    ]
    
    gen = KernelGenerator(seed=42)
    
    for ktype in kernel_types:
        spec = gen.generate(ktype)
        source = gen.to_python_source(spec)
        
        pair = synthesize_pair(source, spec.name, ktype, device="cuda", include_backward=True)
        
        assert pair is not None, f"Failed to synthesize {ktype}"
        assert "cuda_forward" in pair, f"Missing cuda_forward for {ktype}"
        
    print(f"✓ test_all_kernel_types passed ({len(kernel_types)} types)")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("CUDA Extraction Tests (No GPU Required)")
    print("=" * 60)
    
    tests = [
        test_cuda_forward_extraction,
        test_cuda_backward_extraction,
        test_cpu_vs_cuda_difference,
        test_synthesize_pair_cuda,
        test_all_kernel_types,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
