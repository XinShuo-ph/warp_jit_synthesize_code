#!/usr/bin/env python3
"""
Verification script to ensure all components are working correctly.
Run this before using the pipeline in production.
"""

import sys
import os

def check_installation():
    """Check that JAX is properly installed."""
    print("=" * 80)
    print("STEP 1: Checking Installation")
    print("=" * 80)
    
    try:
        import jax
        import jax.numpy as jnp
        print(f"✓ JAX {jax.__version__} installed")
        print(f"✓ Backend: {jax.default_backend()}")
        print(f"✓ Devices: {jax.devices()}")
        return True
    except ImportError as e:
        print(f"✗ JAX not installed: {e}")
        print("  Install with: pip install jax jaxlib")
        return False


def check_ir_extraction():
    """Verify IR extraction works."""
    print("\n" + "=" * 80)
    print("STEP 2: Testing IR Extraction")
    print("=" * 80)
    
    try:
        sys.path.insert(0, '/workspace/jax_jit/code/extraction')
        from ir_extractor import extract_ir
        import jax.numpy as jnp
        
        def test_func(x, y):
            return x + y
        
        x = jnp.array([1., 2., 3.])
        y = jnp.array([4., 5., 6.])
        
        pair = extract_ir(test_func, x, y)
        
        assert pair.function_name == "test_func"
        assert "stablehlo" in pair.stablehlo_ir.lower()
        assert pair.cost_analysis.get('flops', 0) > 0
        
        print("✓ IR extraction working")
        print(f"  Function: {pair.function_name}")
        print(f"  IR lines: {len(pair.stablehlo_ir.splitlines())}")
        print(f"  FLOPs: {pair.cost_analysis.get('flops', 0)}")
        return True
        
    except Exception as e:
        print(f"✗ IR extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_generator():
    """Verify function generator works."""
    print("\n" + "=" * 80)
    print("STEP 3: Testing Function Generator")
    print("=" * 80)
    
    try:
        sys.path.insert(0, '/workspace/jax_jit/code/synthesis')
        from generator import FunctionGenerator, spec_to_callable, generate_example_inputs
        
        gen = FunctionGenerator(seed=42)
        
        # Test each category
        categories = ['arithmetic', 'conditional', 'reduction', 'matrix', 
                     'elementwise', 'broadcasting', 'composite']
        
        for category in categories:
            spec = gen.generate(category)
            func = spec_to_callable(spec)
            inputs = generate_example_inputs(spec.params)
            result = func(*inputs)
            print(f"✓ {category}: {spec.name}")
        
        print(f"✓ All {len(categories)} categories working")
        return True
        
    except Exception as e:
        print(f"✗ Generator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_pipeline():
    """Verify synthesis pipeline works."""
    print("\n" + "=" * 80)
    print("STEP 4: Testing Synthesis Pipeline")
    print("=" * 80)
    
    try:
        sys.path.insert(0, '/workspace/jax_jit/code/synthesis')
        from pipeline import SynthesisPipeline
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = SynthesisPipeline(output_dir=tmpdir, seed=123)
            
            # Generate a few pairs
            pairs = pipeline.generate_batch(count=5, save=True, verbose=False)
            
            assert len(pairs) == 5
            
            # Validate one
            is_valid = pipeline.validate_pair(pairs[0])
            assert is_valid
            
            # Get stats
            stats = pipeline.get_statistics()
            assert stats['total_pairs'] == 5
            
            print("✓ Pipeline working")
            print(f"  Generated: {len(pairs)} pairs")
            print(f"  Valid: {is_valid}")
            print(f"  Avg FLOPs: {stats['avg_flops']:.1f}")
            return True
            
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset():
    """Check that dataset exists and is valid."""
    print("\n" + "=" * 80)
    print("STEP 5: Checking Dataset")
    print("=" * 80)
    
    try:
        import json
        from pathlib import Path
        
        data_dir = Path("/workspace/jax_jit/data/samples")
        json_files = list(data_dir.glob("*.json"))
        
        print(f"✓ Found {len(json_files)} JSON files")
        
        if json_files:
            # Load one to verify format
            with open(json_files[0]) as f:
                sample = json.load(f)
            
            required_fields = ['function_name', 'python_source', 'stablehlo_ir', 'cost_analysis']
            for field in required_fields:
                assert field in sample, f"Missing field: {field}"
            
            print(f"✓ Dataset format valid")
            print(f"  Sample: {sample['function_name']}")
            print(f"  Category: {sample.get('category', 'unknown')}")
        else:
            print("⚠ No dataset files found (generate with batch_generator.py)")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_tests():
    """Run the test suite."""
    print("\n" + "=" * 80)
    print("STEP 6: Running Test Suite")
    print("=" * 80)
    
    try:
        sys.path.insert(0, '/workspace/jax_jit/code/extraction')
        from test_ir_extractor import run_all_tests
        
        success = run_all_tests()
        
        if success:
            print("✓ All tests passed")
            return True
        else:
            print("✗ Some tests failed")
            return False
            
    except Exception as e:
        print(f"✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("JAX JIT CODE SYNTHESIS - VERIFICATION")
    print("=" * 80)
    print("\nThis script verifies that all components are working correctly.\n")
    
    checks = [
        ("Installation", check_installation),
        ("IR Extraction", check_ir_extraction),
        ("Function Generator", check_generator),
        ("Synthesis Pipeline", check_pipeline),
        ("Dataset", check_dataset),
        ("Test Suite", check_tests),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "=" * 80)
    print(f"Result: {passed}/{total} checks passed")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 All checks passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("  1. Generate a dataset: python3 code/synthesis/batch_generator.py --count 1000")
        print("  2. Validate quality: python3 code/synthesis/validate_dataset.py")
        print("  3. Run the demo: python3 demo.py")
        return True
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
