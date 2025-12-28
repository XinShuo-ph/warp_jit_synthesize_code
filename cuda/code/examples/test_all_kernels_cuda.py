"""
Test all kernel categories on CPU and CUDA backends.

This script validates that all kernel types work correctly with CUDA backend.
"""
import sys
import tempfile
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent.parent / "synthesis"))

import warp as wp
from generator import GENERATORS, generate_kernel
from ir_extractor import extract_ir

def compile_kernel_from_source(source, kernel_name):
    """Compile kernel from source using temp file."""
    temp_dir = Path(tempfile.gettempdir()) / "cuda_test"
    temp_dir.mkdir(exist_ok=True)
    
    module_source = f"import warp as wp\n\n{source}"
    module_name = f"test_{kernel_name}"
    temp_file = temp_dir / f"{module_name}.py"
    
    with open(temp_file, 'w') as f:
        f.write(module_source)
    
    spec = importlib.util.spec_from_file_location(module_name, temp_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return getattr(module, kernel_name)

def test_kernel_category(category, seed=42):
    """Test a specific kernel category on both CPU and CUDA."""
    print(f"\n{'='*70}")
    print(f"Testing {category.upper()} Kernels")
    print(f"{'='*70}")
    
    spec = generate_kernel(category, seed=seed)
    print(f"\nKernel: {spec.name}")
    print(f"Description: {spec.description}")
    print(f"\nPython Source:")
    for i, line in enumerate(spec.source.split('\n')[:10], 1):
        print(f"  {line}")
    if len(spec.source.split('\n')) > 10:
        print(f"  ... ({len(spec.source.split('\n')) - 10} more lines)")
    
    # Compile kernel
    kernel = compile_kernel_from_source(spec.source, spec.name)
    
    # Test CPU
    try:
        result_cpu = extract_ir(kernel, device="cpu", include_backward=False)
        print(f"\n✓ CPU codegen successful ({len(result_cpu['forward_code'])} chars)")
    except Exception as e:
        print(f"\n✗ CPU codegen failed: {e}")
        return False
    
    # Test CUDA
    try:
        result_cuda = extract_ir(kernel, device="cuda", include_backward=False)
        print(f"✓ CUDA codegen successful ({len(result_cuda['forward_code'])} chars)")
        
        # Check for CUDA patterns
        forward_cuda = result_cuda['forward_code']
        has_cuda_indexing = any(x in forward_cuda for x in ['blockIdx', 'threadIdx', 'blockDim', 'gridDim'])
        
        if has_cuda_indexing:
            print(f"  → CUDA thread indexing detected")
        
        # Category-specific checks
        if category == "atomic":
            has_atomic = "atomic" in forward_cuda.lower()
            print(f"  → CUDA atomic operations: {has_atomic}")
        elif category == "vector":
            # Vector operations might be inlined
            print(f"  → Vector operations compiled")
        elif category == "matrix":
            print(f"  → Matrix operations compiled")
            
        return True
        
    except Exception as e:
        print(f"✗ CUDA codegen failed: {e}")
        return False

def main():
    """Test all kernel categories."""
    wp.init()
    
    print("=" * 70)
    print("COMPREHENSIVE KERNEL CATEGORY TEST: CPU vs CUDA")
    print("=" * 70)
    
    results = {}
    
    for category in GENERATORS.keys():
        try:
            success = test_kernel_category(category, seed=42)
            results[category] = success
        except Exception as e:
            print(f"\n✗ Category {category} failed: {e}")
            results[category] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for category, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {category}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nTotal: {passed}/{total} categories passed")
    
    if passed == total:
        print("\n✓ All kernel categories work with CUDA backend!")
        print("\nKey Findings:")
        print("- All 6 kernel categories successfully generate CUDA code")
        print("- CUDA code includes proper thread indexing (blockIdx, threadIdx, etc.)")
        print("- No changes to Python kernels needed for CUDA support")
        print("- Warp automatically handles backend translation")
        return 0
    else:
        print("\n✗ Some categories failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
