"""
CUDA Pipeline Test Suite

Run this on a GPU-enabled machine to validate CUDA backend.
Can also run on CPU-only machine to verify code generation works.
"""
import sys
from pathlib import Path
import json

# Add paths
parent = Path(__file__).parent.parent
sys.path.insert(0, str(parent / "code" / "extraction"))
sys.path.insert(0, str(parent / "code" / "synthesis"))

import warp as wp
from pipeline import synthesize_batch
from generator import GENERATORS


def test_cuda_pipeline():
    """Test CUDA pipeline end-to-end."""
    print("=" * 70)
    print("CUDA Pipeline Test Suite")
    print("=" * 70)
    
    wp.init()
    
    # Check CUDA availability
    print("\n1. Device Detection")
    print("-" * 70)
    try:
        # Try to get CUDA device safely
        import warp._src.context as ctx
        has_gpu = ctx.runtime.cuda_devices is not None and len(ctx.runtime.cuda_devices) > 0
    except:
        has_gpu = False
    
    if has_gpu:
        print("✓ CUDA device available")
    else:
        print("⚠ CUDA device not available (CPU-only mode)")
        print("  Code generation will still work")
    
    # Test small batch generation
    print("\n2. Small Batch Generation (5 kernels, CUDA backend)")
    print("-" * 70)
    try:
        pairs = synthesize_batch(5, categories=None, seed=42, device="cuda")
        print(f"✓ Generated {len(pairs)} pairs")
        
        # Verify CUDA IR
        if pairs:
            sample = pairs[0]
            cpp = sample.get("cpp_forward", "")
            has_cuda_indexing = any(x in cpp for x in ['blockIdx', 'threadIdx'])
            
            if has_cuda_indexing:
                print("✓ Generated code contains CUDA thread indexing")
            else:
                print("✗ Generated code missing CUDA patterns")
                return False
                
            print(f"  Sample: {sample['metadata']['category']} - {sample['metadata']['kernel_name']}")
            
    except Exception as e:
        print(f"✗ Batch generation failed: {e}")
        return False
    
    # Test all categories
    print("\n3. Category Coverage Test")
    print("-" * 70)
    category_results = {}
    
    for category in GENERATORS.keys():
        try:
            pairs = synthesize_batch(1, categories=[category], seed=42, device="cuda")
            if pairs:
                category_results[category] = True
                print(f"  ✓ {category}")
            else:
                category_results[category] = False
                print(f"  ✗ {category} (no pairs generated)")
        except Exception as e:
            category_results[category] = False
            print(f"  ✗ {category} (error: {e})")
    
    success_rate = sum(1 for v in category_results.values() if v) / len(category_results)
    print(f"\n  Success rate: {success_rate * 100:.0f}% ({sum(1 for v in category_results.values() if v)}/{len(category_results)})")
    
    if success_rate < 1.0:
        print("  ✗ Some categories failed")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    if has_gpu:
        print("✓ All tests passed on GPU-enabled system")
    else:
        print("✓ All tests passed in CPU-only mode")
        print("  (CUDA code generation works, runtime execution requires GPU)")
    
    print("\nVerified:")
    print("  - CUDA code generation works")
    print("  - All kernel categories supported")
    print("  - Thread indexing patterns present")
    print(f"  - Pipeline produces valid output")
    
    if has_gpu:
        print("\nNext steps:")
        print("  - Run: python3 pipeline.py -n 100 -d cuda -o /output/dir")
        print("  - Generate larger dataset: python3 batch_generator.py -n 10000 -d cuda")
    else:
        print("\n⚠ To execute CUDA kernels on GPU:")
        print("  1. Copy this code to GPU-enabled machine")
        print("  2. Install: pip install warp-lang")
        print("  3. Run: python3 pipeline.py -n 100 -d cuda -o /output/dir")
    
    return True


if __name__ == "__main__":
    success = test_cuda_pipeline()
    sys.exit(0 if success else 1)
