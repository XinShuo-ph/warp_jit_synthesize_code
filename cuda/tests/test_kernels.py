"""
Test individual kernel types for CUDA code generation.

These tests verify each kernel category generates valid CUDA code.
No GPU execution - only code generation is tested.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "synthesis"))

import warp as wp
wp.init()

from generator import KernelGenerator
from pipeline import compile_kernel_from_source, extract_ir_from_kernel


def verify_cuda_kernel(ktype: str, seed: int = 42) -> dict:
    """Generate and verify a CUDA kernel of the given type."""
    gen = KernelGenerator(seed=seed)
    spec = gen.generate(ktype)
    source = gen.to_python_source(spec)
    
    kernel = compile_kernel_from_source(source, spec.name)
    ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=True)
    
    return {
        "type": ktype,
        "name": spec.name,
        "source": source,
        "has_forward": ir["forward_code"] is not None,
        "has_backward": ir["backward_code"] is not None,
        "forward_code": ir["forward_code"],
        "backward_code": ir["backward_code"],
    }


def test_arithmetic_cuda():
    """Test arithmetic kernel CUDA generation."""
    result = verify_cuda_kernel("arithmetic")
    assert result["has_forward"], "Arithmetic: missing forward"
    assert result["has_backward"], "Arithmetic: missing backward"
    assert "wp::add" in result["forward_code"] or "wp::mul" in result["forward_code"]
    print("✓ test_arithmetic_cuda passed")


def test_conditional_cuda():
    """Test conditional kernel CUDA generation."""
    result = verify_cuda_kernel("conditional")
    assert result["has_forward"], "Conditional: missing forward"
    assert result["has_backward"], "Conditional: missing backward"
    # Conditional code should have comparison
    print("✓ test_conditional_cuda passed")


def test_loop_cuda():
    """Test loop kernel CUDA generation."""
    result = verify_cuda_kernel("loop")
    assert result["has_forward"], "Loop: missing forward"
    assert result["has_backward"], "Loop: missing backward"
    print("✓ test_loop_cuda passed")


def test_math_cuda():
    """Test math kernel CUDA generation."""
    result = verify_cuda_kernel("math")
    assert result["has_forward"], "Math: missing forward"
    assert result["has_backward"], "Math: missing backward"
    # Should contain math functions
    code = result["forward_code"]
    has_math = any(f in code for f in ["wp::sin", "wp::cos", "wp::exp", "wp::sqrt", "wp::abs", "wp::log"])
    assert has_math, "Math kernel should contain math functions"
    print("✓ test_math_cuda passed")


def test_vector_cuda():
    """Test vector kernel CUDA generation."""
    result = verify_cuda_kernel("vector")
    assert result["has_forward"], "Vector: missing forward"
    assert result["has_backward"], "Vector: missing backward"
    # Should reference vec3 types
    assert "vec3" in result["forward_code"].lower() or "wp::vec" in result["forward_code"]
    print("✓ test_vector_cuda passed")


def test_atomic_cuda():
    """Test atomic kernel CUDA generation."""
    result = verify_cuda_kernel("atomic")
    assert result["has_forward"], "Atomic: missing forward"
    assert result["has_backward"], "Atomic: missing backward"
    assert "atomic" in result["forward_code"].lower()
    print("✓ test_atomic_cuda passed")


def test_nested_cuda():
    """Test nested loop kernel CUDA generation."""
    result = verify_cuda_kernel("nested")
    assert result["has_forward"], "Nested: missing forward"
    assert result["has_backward"], "Nested: missing backward"
    print("✓ test_nested_cuda passed")


def test_multi_cond_cuda():
    """Test multi-conditional kernel CUDA generation."""
    result = verify_cuda_kernel("multi_cond")
    assert result["has_forward"], "Multi-cond: missing forward"
    assert result["has_backward"], "Multi-cond: missing backward"
    print("✓ test_multi_cond_cuda passed")


def test_combined_cuda():
    """Test combined kernel CUDA generation."""
    result = verify_cuda_kernel("combined")
    assert result["has_forward"], "Combined: missing forward"
    assert result["has_backward"], "Combined: missing backward"
    print("✓ test_combined_cuda passed")


def test_scalar_param_cuda():
    """Test scalar parameter kernel CUDA generation."""
    result = verify_cuda_kernel("scalar_param")
    assert result["has_forward"], "Scalar param: missing forward"
    assert result["has_backward"], "Scalar param: missing backward"
    print("✓ test_scalar_param_cuda passed")


def test_random_math_cuda():
    """Test random math kernel CUDA generation."""
    result = verify_cuda_kernel("random_math")
    assert result["has_forward"], "Random math: missing forward"
    assert result["has_backward"], "Random math: missing backward"
    print("✓ test_random_math_cuda passed")


def run_all_tests():
    """Run all kernel type tests."""
    print("=" * 60)
    print("CUDA Kernel Type Tests (No GPU Required)")
    print("=" * 60)
    
    tests = [
        test_arithmetic_cuda,
        test_conditional_cuda,
        test_loop_cuda,
        test_math_cuda,
        test_vector_cuda,
        test_atomic_cuda,
        test_nested_cuda,
        test_multi_cond_cuda,
        test_combined_cuda,
        test_scalar_param_cuda,
        test_random_math_cuda,
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
