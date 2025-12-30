#!/usr/bin/env python3
"""
Test cases for IR Extractor
Tests various function types to ensure IR extraction works correctly
"""
import jax.numpy as jnp
from ir_extractor import IRExtractor
import json


def test_arithmetic():
    """Test basic arithmetic operations"""
    print("\n" + "=" * 70)
    print("TEST 1: Arithmetic Operations")
    print("=" * 70)
    
    extractor = IRExtractor(ir_type="both")
    
    # Test functions
    test_cases = [
        ("add", lambda x, y: x + y, [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]),
        ("subtract", lambda x, y: x - y, [jnp.array([5.0, 6.0]), jnp.array([1.0, 2.0])]),
        ("multiply", lambda x, y: x * y, [jnp.array([2.0, 3.0]), jnp.array([4.0, 5.0])]),
        ("divide", lambda x, y: x / y, [jnp.array([10.0, 20.0]), jnp.array([2.0, 5.0])]),
    ]
    
    pairs = []
    for name, func, inputs in test_cases:
        ir = extractor.extract(func, *inputs)
        print(f"   {name:12s}: Jaxpr={len(ir['jaxpr']):4d} chars, "
              f"StableHLO={len(ir['stablehlo']):4d} chars")
        
        pair = extractor.create_training_pair(
            python_code=f"lambda x, y: x {name[0]} y",
            func=func,
            example_inputs=inputs,
            metadata={"category": "arithmetic", "operation": name}
        )
        pairs.append(pair)
    
    print(f"   ✓ Created {len(pairs)} training pairs")
    return pairs


def test_array_operations():
    """Test array manipulation operations"""
    print("\n" + "=" * 70)
    print("TEST 2: Array Operations")
    print("=" * 70)
    
    extractor = IRExtractor(ir_type="both")
    
    test_cases = [
        ("reshape", lambda x: jnp.reshape(x, (2, 3)), [jnp.arange(6.0)]),
        ("transpose", lambda x: x.T, [jnp.ones((3, 4))]),
        ("slice", lambda x: x[1:3], [jnp.arange(5.0)]),
        ("concatenate", lambda x, y: jnp.concatenate([x, y]), 
         [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]),
    ]
    
    pairs = []
    for name, func, inputs in test_cases:
        ir = extractor.extract(func, *inputs)
        print(f"   {name:12s}: Jaxpr={len(ir['jaxpr']):4d} chars, "
              f"StableHLO={len(ir['stablehlo']):4d} chars")
        
        pair = extractor.create_training_pair(
            python_code=f"# {name} operation",
            func=func,
            example_inputs=inputs,
            metadata={"category": "array", "operation": name}
        )
        pairs.append(pair)
    
    print(f"   ✓ Created {len(pairs)} training pairs")
    return pairs


def test_math_functions():
    """Test mathematical functions"""
    print("\n" + "=" * 70)
    print("TEST 3: Math Functions")
    print("=" * 70)
    
    extractor = IRExtractor(ir_type="both")
    x = jnp.array([0.0, 1.0, 2.0])
    
    test_cases = [
        ("sin", lambda x: jnp.sin(x)),
        ("cos", lambda x: jnp.cos(x)),
        ("exp", lambda x: jnp.exp(x)),
        ("log", lambda x: jnp.log(x + 1.0)),  # +1 to avoid log(0)
        ("sqrt", lambda x: jnp.sqrt(jnp.abs(x))),
        ("tanh", lambda x: jnp.tanh(x)),
    ]
    
    pairs = []
    for name, func in test_cases:
        ir = extractor.extract(func, x)
        print(f"   {name:12s}: Jaxpr={len(ir['jaxpr']):4d} chars, "
              f"StableHLO={len(ir['stablehlo']):4d} chars")
        
        pair = extractor.create_training_pair(
            python_code=f"lambda x: jnp.{name}(x)",
            func=func,
            example_inputs=[x],
            metadata={"category": "math", "function": name}
        )
        pairs.append(pair)
    
    print(f"   ✓ Created {len(pairs)} training pairs")
    return pairs


def test_reductions():
    """Test reduction operations"""
    print("\n" + "=" * 70)
    print("TEST 4: Reduction Operations")
    print("=" * 70)
    
    extractor = IRExtractor(ir_type="both")
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    test_cases = [
        ("sum", lambda x: jnp.sum(x)),
        ("mean", lambda x: jnp.mean(x)),
        ("max", lambda x: jnp.max(x)),
        ("min", lambda x: jnp.min(x)),
        ("sum_axis0", lambda x: jnp.sum(x, axis=0)),
        ("mean_axis1", lambda x: jnp.mean(x, axis=1)),
    ]
    
    pairs = []
    for name, func in test_cases:
        ir = extractor.extract(func, x)
        print(f"   {name:12s}: Jaxpr={len(ir['jaxpr']):4d} chars, "
              f"StableHLO={len(ir['stablehlo']):4d} chars")
        
        pair = extractor.create_training_pair(
            python_code=f"# {name}",
            func=func,
            example_inputs=[x],
            metadata={"category": "reduction", "operation": name}
        )
        pairs.append(pair)
    
    print(f"   ✓ Created {len(pairs)} training pairs")
    return pairs


def test_linear_algebra():
    """Test linear algebra operations"""
    print("\n" + "=" * 70)
    print("TEST 5: Linear Algebra")
    print("=" * 70)
    
    extractor = IRExtractor(ir_type="both")
    
    test_cases = [
        ("dot", lambda x, y: jnp.dot(x, y), 
         [jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])]),
        ("matmul", lambda A, B: jnp.matmul(A, B),
         [jnp.ones((2, 3)), jnp.ones((3, 4))]),
        ("outer", lambda x, y: jnp.outer(x, y),
         [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0, 5.0])]),
    ]
    
    pairs = []
    for name, func, inputs in test_cases:
        ir = extractor.extract(func, *inputs)
        print(f"   {name:12s}: Jaxpr={len(ir['jaxpr']):4d} chars, "
              f"StableHLO={len(ir['stablehlo']):4d} chars")
        
        pair = extractor.create_training_pair(
            python_code=f"# {name}",
            func=func,
            example_inputs=inputs,
            metadata={"category": "linalg", "operation": name}
        )
        pairs.append(pair)
    
    print(f"   ✓ Created {len(pairs)} training pairs")
    return pairs


def main():
    """Run all tests and save results"""
    print("=" * 70)
    print("IR Extractor Test Suite")
    print("=" * 70)
    
    all_pairs = []
    
    # Run all test categories
    all_pairs.extend(test_arithmetic())
    all_pairs.extend(test_array_operations())
    all_pairs.extend(test_math_functions())
    all_pairs.extend(test_reductions())
    all_pairs.extend(test_linear_algebra())
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"   Total training pairs created: {len(all_pairs)}")
    
    # Save to file
    extractor = IRExtractor()
    import os
    os.makedirs("../../data/samples", exist_ok=True)
    output_path = "../../data/samples/test_cases.json"
    extractor.save_pairs(all_pairs, output_path)
    print(f"   Saved to: {output_path}")
    
    # Verify we can load it back
    with open(output_path, 'r') as f:
        loaded = json.load(f)
    print(f"   Verified: Successfully loaded {len(loaded)} pairs")
    
    print("\n" + "=" * 70)
    print("SUCCESS: All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
