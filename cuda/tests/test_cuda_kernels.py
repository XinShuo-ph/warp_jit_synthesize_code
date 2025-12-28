"""
Test CUDA kernel generation and structure validation.

This test validates the generated CUDA code structure without requiring a GPU.
For actual GPU execution tests, see run_on_gpu.py.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "code/extraction"))
sys.path.insert(0, str(Path(__file__).parent.parent / "code/synthesis"))

import warp as wp
from cuda_pipeline import run_cuda_pipeline
from generator import GENERATORS


def test_cuda_structure():
    """Test that generated CUDA code has expected structure."""
    print("=" * 70)
    print("Test: CUDA Code Structure Validation")
    print("=" * 70)
    
    # Generate a small batch
    output_dir = "/tmp/cuda_test_structure"
    pairs = run_cuda_pipeline(n=6, output_dir=output_dir, seed=999, device="cuda")
    
    assert len(pairs) == 6, f"Expected 6 pairs, got {len(pairs)}"
    print(f"✓ Generated {len(pairs)} pairs")
    
    # Check each pair
    for i, pair in enumerate(pairs):
        cuda_code = pair["cuda_forward"]
        metadata = pair["metadata"]
        
        # Verify metadata
        assert metadata["device"] == "cuda", f"Pair {i}: wrong device"
        assert "category" in metadata, f"Pair {i}: missing category"
        
        # Verify CUDA code structure
        assert "extern" in cuda_code, f"Pair {i}: missing extern C"
        assert "__global__" in cuda_code, f"Pair {i}: missing __global__"
        assert "cuda_kernel_forward" in cuda_code, f"Pair {i}: wrong kernel name"
        assert "blockDim" in cuda_code, f"Pair {i}: missing grid-stride loop"
        assert "blockIdx" in cuda_code, f"Pair {i}: missing block index"
        assert "threadIdx" in cuda_code, f"Pair {i}: missing thread index"
        assert "tile_shared_storage_t" in cuda_code, f"Pair {i}: missing shared memory"
        
        print(f"  ✓ Pair {i} ({metadata['category']}): valid CUDA structure")
    
    print("\n✓ All structure checks passed")
    return True


def test_all_categories():
    """Test that all kernel categories generate valid CUDA."""
    print("\n" + "=" * 70)
    print("Test: All Kernel Categories")
    print("=" * 70)
    
    results = {}
    
    for category in GENERATORS.keys():
        print(f"\nTesting {category}...")
        
        try:
            pairs = run_cuda_pipeline(
                n=2,
                output_dir=f"/tmp/cuda_test_{category}",
                categories=[category],
                seed=2000,
                device="cuda"
            )
            
            assert len(pairs) == 2, f"{category}: expected 2 pairs, got {len(pairs)}"
            
            # Check CUDA structure
            for pair in pairs:
                assert "__global__" in pair["cuda_forward"], f"{category}: missing __global__"
            
            results[category] = "✓ PASS"
            print(f"  ✓ {category}: generated valid CUDA")
            
        except Exception as e:
            results[category] = f"✗ FAIL: {e}"
            print(f"  ✗ {category}: {e}")
    
    print("\n" + "=" * 70)
    print("Category Results:")
    print("=" * 70)
    for cat, result in results.items():
        print(f"  {cat:15s}: {result}")
    
    # Check all passed
    passed = sum(1 for r in results.values() if r.startswith("✓"))
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    assert passed == total, "Some categories failed"
    
    return True


def test_cuda_vs_cpu():
    """Test that same kernel generates different code for CUDA vs CPU."""
    print("\n" + "=" * 70)
    print("Test: CUDA vs CPU Code Generation")
    print("=" * 70)
    
    # Generate same kernel for both devices
    cpu_pairs = run_cuda_pipeline(
        n=1,
        output_dir="/tmp/cuda_test_cpu",
        seed=3000,
        device="cpu"
    )
    
    cuda_pairs = run_cuda_pipeline(
        n=1,
        output_dir="/tmp/cuda_test_cuda",
        seed=3000,
        device="cuda"
    )
    
    assert len(cpu_pairs) == 1, "CPU generation failed"
    assert len(cuda_pairs) == 1, "CUDA generation failed"
    
    cpu_code = cpu_pairs[0]["cuda_forward"]  # Note: still stored in 'cuda_forward' key
    cuda_code = cuda_pairs[0]["cuda_forward"]
    
    # Check they're different
    assert cpu_code != cuda_code, "CPU and CUDA code should differ"
    print("  ✓ CPU and CUDA generate different code")
    
    # Check CPU has cpu_kernel_forward
    assert "cpu_kernel_forward" in cpu_code, "CPU code missing cpu_kernel"
    print("  ✓ CPU code has cpu_kernel_forward")
    
    # Check CUDA has cuda_kernel_forward and __global__
    assert "cuda_kernel_forward" in cuda_code, "CUDA code missing cuda_kernel"
    assert "__global__" in cuda_code, "CUDA code missing __global__"
    print("  ✓ CUDA code has cuda_kernel_forward and __global__")
    
    # Check CPU uses task_index
    assert "task_index" in cpu_code, "CPU code missing task_index"
    print("  ✓ CPU code uses task_index")
    
    # Check CUDA uses blockIdx/threadIdx
    assert "blockIdx" in cuda_code and "threadIdx" in cuda_code, "CUDA code missing thread indices"
    print("  ✓ CUDA code uses blockIdx/threadIdx")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CUDA Kernel Structure Tests")
    print("(These tests validate code generation, not GPU execution)")
    print("=" * 70)
    
    try:
        test_cuda_structure()
        test_all_categories()
        test_cuda_vs_cpu()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nNote: These tests validate CUDA code structure.")
        print("To test actual GPU execution, run: python3 tests/run_on_gpu.py")
        print("(Requires CUDA-capable GPU)")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
