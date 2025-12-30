#!/usr/bin/env python3
"""
Comprehensive demo of the JAX JIT Code Synthesis Pipeline.
Run this script to see all components in action.
"""

import sys
sys.path.insert(0, '/workspace/jax_jit/code/extraction')
sys.path.insert(0, '/workspace/jax_jit/code/synthesis')

import jax.numpy as jnp
from ir_extractor import extract_ir, extract_with_grad
from generator import FunctionGenerator, spec_to_code, spec_to_callable, generate_example_inputs
from pipeline import SynthesisPipeline
from batch_generator import BatchGenerator


def demo_1_basic_extraction():
    """Demo 1: Extract IR from a simple function."""
    print("=" * 80)
    print("DEMO 1: Basic IR Extraction")
    print("=" * 80)
    
    # Define a simple function
    def saxpy(alpha, x, y):
        """Compute alpha * x + y (BLAS operation)."""
        return alpha * x + y
    
    # Create example inputs
    alpha = 2.5
    x = jnp.array([1., 2., 3., 4.])
    y = jnp.array([5., 6., 7., 8.])
    
    # Extract IR
    pair = extract_ir(saxpy, alpha, x, y)
    
    print(f"\nFunction: {pair.function_name}")
    print(f"Cost Analysis: {pair.cost_analysis}")
    
    print("\nPython Source:")
    print(pair.python_source)
    
    print("\nStableHLO IR:")
    print(pair.stablehlo_ir)
    
    # Execute to verify
    result = saxpy(alpha, x, y)
    print(f"\nExecution result: {result}")
    print("✓ Demo 1 complete\n")


def demo_2_gradient_extraction():
    """Demo 2: Extract forward and backward pass IR."""
    print("=" * 80)
    print("DEMO 2: Gradient Extraction")
    print("=" * 80)
    
    def loss_function(params, data):
        """Simple L2 loss."""
        prediction = params * data
        return jnp.sum((prediction - data) ** 2)
    
    params = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([4.0, 5.0, 6.0])
    
    # Extract forward and backward
    fwd_pair, bwd_pair = extract_with_grad(loss_function, params, data)
    
    print(f"\nForward Pass:")
    print(f"  Function: {fwd_pair.function_name}")
    print(f"  IR lines: {len(fwd_pair.stablehlo_ir.splitlines())}")
    print(f"  FLOPs: {fwd_pair.cost_analysis.get('flops', 0)}")
    
    print(f"\nBackward Pass (Gradient):")
    print(f"  Function: {bwd_pair.function_name}")
    print(f"  IR lines: {len(bwd_pair.stablehlo_ir.splitlines())}")
    print(f"  FLOPs: {bwd_pair.cost_analysis.get('flops', 0)}")
    
    print("\n✓ Demo 2 complete\n")


def demo_3_function_generation():
    """Demo 3: Generate random functions."""
    print("=" * 80)
    print("DEMO 3: Programmatic Function Generation")
    print("=" * 80)
    
    generator = FunctionGenerator(seed=42)
    
    categories = ['arithmetic', 'conditional', 'reduction', 'matrix']
    
    for category in categories:
        spec = generator.generate(category)
        code = spec_to_code(spec)
        
        print(f"\nCategory: {category}")
        print(f"Function: {spec.name}")
        print("Generated Code:")
        print(code)
        
        # Convert to callable and execute
        func = spec_to_callable(spec)
        inputs = generate_example_inputs(spec.params)
        result = func(*inputs)
        
        print(f"Execution successful: result shape = {result.shape if hasattr(result, 'shape') else type(result)}")
    
    print("\n✓ Demo 3 complete\n")


def demo_4_pipeline():
    """Demo 4: End-to-end synthesis pipeline."""
    print("=" * 80)
    print("DEMO 4: Synthesis Pipeline")
    print("=" * 80)
    
    pipeline = SynthesisPipeline(output_dir="/tmp/jax_demo_samples", seed=123)
    
    print("\nGenerating 10 pairs across all categories...")
    pairs = pipeline.generate_batch(count=10, save=True, verbose=False)
    
    print(f"\nGenerated {len(pairs)} pairs")
    
    # Show statistics
    stats = pipeline.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Categories: {stats['categories']}")
    print(f"  Avg Python lines: {stats['avg_python_lines']:.1f}")
    print(f"  Avg IR lines: {stats['avg_ir_lines']:.1f}")
    print(f"  Avg FLOPs: {stats['avg_flops']:.1f}")
    
    # Validate one pair
    print("\nValidating a sample pair...")
    sample_pair = pairs[0]
    is_valid = pipeline.validate_pair(sample_pair)
    print(f"  Function: {sample_pair['function_name']}")
    print(f"  Valid: {is_valid}")
    
    print("\n✓ Demo 4 complete\n")


def demo_5_batch_generation():
    """Demo 5: High-throughput batch generation."""
    print("=" * 80)
    print("DEMO 5: Batch Generation")
    print("=" * 80)
    
    generator = BatchGenerator(output_dir="/tmp/jax_demo_batch", seed=456, verbose=True)
    
    print("\nGenerating 50 pairs...")
    stats = generator.generate_dataset(target_count=50, batch_size=25)
    
    print(f"\nFinal Statistics:")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Generation time: {stats['total_time']:.2f}s")
    print(f"  Rate: {stats['rate']:.1f} pairs/sec")
    
    print("\n✓ Demo 5 complete\n")


def demo_6_real_world_example():
    """Demo 6: Extract IR from a real-world function."""
    print("=" * 80)
    print("DEMO 6: Real-World Example - Softmax")
    print("=" * 80)
    
    def softmax(logits):
        """Compute softmax activation."""
        exp_logits = jnp.exp(logits - jnp.max(logits))
        return exp_logits / jnp.sum(exp_logits)
    
    logits = jnp.array([2.0, 1.0, 0.1, -1.0])
    
    pair = extract_ir(softmax, logits)
    
    print("\nFunction: Softmax")
    print(f"Python lines: {len(pair.python_source.splitlines())}")
    print(f"IR lines: {len(pair.stablehlo_ir.splitlines())}")
    print(f"FLOPs: {pair.cost_analysis.get('flops', 0)}")
    print(f"Transcendentals: {pair.cost_analysis.get('transcendentals', 0)}")
    
    print("\nFirst 15 lines of StableHLO IR:")
    for i, line in enumerate(pair.stablehlo_ir.splitlines()[:15], 1):
        print(f"  {i:2d}: {line}")
    
    # Verify execution
    result = softmax(logits)
    print(f"\nSoftmax result: {result}")
    print(f"Sum of probabilities: {jnp.sum(result):.6f} (should be 1.0)")
    
    print("\n✓ Demo 6 complete\n")


def demo_7_comparison():
    """Demo 7: Compare different implementations."""
    print("=" * 80)
    print("DEMO 7: Implementation Comparison")
    print("=" * 80)
    
    # Two ways to compute dot product
    def dot_v1(x, y):
        """Dot product using jnp.dot."""
        return jnp.dot(x, y)
    
    def dot_v2(x, y):
        """Dot product using explicit sum."""
        return jnp.sum(x * y)
    
    x = jnp.array([1., 2., 3., 4.])
    y = jnp.array([5., 6., 7., 8.])
    
    pair1 = extract_ir(dot_v1, x, y)
    pair2 = extract_ir(dot_v2, x, y)
    
    print("\nImplementation 1: jnp.dot")
    print(f"  IR lines: {len(pair1.stablehlo_ir.splitlines())}")
    print(f"  FLOPs: {pair1.cost_analysis.get('flops', 0)}")
    print(f"  Uses: {'dot_general' if 'dot_general' in pair1.stablehlo_ir else 'other ops'}")
    
    print("\nImplementation 2: jnp.sum(x * y)")
    print(f"  IR lines: {len(pair2.stablehlo_ir.splitlines())}")
    print(f"  FLOPs: {pair2.cost_analysis.get('flops', 0)}")
    print(f"  Uses: {'reduce' if 'reduce' in pair2.stablehlo_ir else 'other ops'}")
    
    # Verify both produce same result
    result1 = dot_v1(x, y)
    result2 = dot_v2(x, y)
    print(f"\nResults match: {jnp.allclose(result1, result2)}")
    
    print("\n✓ Demo 7 complete\n")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("JAX JIT CODE SYNTHESIS - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("\nThis demo showcases all components of the pipeline.")
    print("Each demo is self-contained and demonstrates a key feature.\n")
    
    demos = [
        demo_1_basic_extraction,
        demo_2_gradient_extraction,
        demo_3_function_generation,
        demo_4_pipeline,
        demo_5_batch_generation,
        demo_6_real_world_example,
        demo_7_comparison,
    ]
    
    for i, demo in enumerate(demos, 1):
        print(f"\nRunning Demo {i}/{len(demos)}...")
        try:
            demo()
        except Exception as e:
            print(f"✗ Demo {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 80)
    print("ALL DEMOS COMPLETE")
    print("=" * 80)
    print("\nYou can now:")
    print("  1. Generate large datasets: python3 code/synthesis/batch_generator.py")
    print("  2. Validate quality: python3 code/synthesis/validate_dataset.py")
    print("  3. Extract IR from your own functions using the examples above")
    print("\nSee README.md and QUICKSTART.md for more information.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
