#!/usr/bin/env python3
"""
IR Extraction Example: Demonstrates extracting Jaxpr and HLO from JAX functions
"""
import jax
import jax.numpy as jnp
from jax import make_jaxpr, jit


def simple_add(x, y):
    """Simple addition function"""
    return x + y


def complex_function(x):
    """More complex function with multiple operations"""
    y = x * 2
    z = jnp.sin(y)
    return jnp.sum(z ** 2)


def matrix_multiply(A, B):
    """Matrix multiplication"""
    return jnp.matmul(A, B)


def main():
    print("=" * 70)
    print("JAX IR Extraction Demo")
    print("=" * 70)
    
    # Example 1: Simple addition - extract Jaxpr
    print("\n" + "-" * 70)
    print("1. JAXPR EXTRACTION: Simple Addition")
    print("-" * 70)
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    
    jaxpr = make_jaxpr(simple_add)(x, y)
    print(f"\nFunction: simple_add(x, y) = x + y")
    print(f"Input shapes: x={x.shape}, y={y.shape}")
    print(f"\nJaxpr representation:")
    print(jaxpr)
    
    # Example 2: Simple addition - extract HLO
    print("\n" + "-" * 70)
    print("2. HLO/StableHLO EXTRACTION: Simple Addition")
    print("-" * 70)
    lowered = jit(simple_add).lower(x, y)
    hlo_text = lowered.as_text()
    print(f"\nStableHLO Text (truncated to first 800 chars):")
    print(hlo_text[:800])
    if len(hlo_text) > 800:
        print("...")
    
    # Example 3: Complex function - extract Jaxpr
    print("\n" + "-" * 70)
    print("3. JAXPR EXTRACTION: Complex Function")
    print("-" * 70)
    x = jnp.array([1.0, 2.0, 3.0])
    jaxpr_complex = make_jaxpr(complex_function)(x)
    print(f"\nFunction: complex_function(x)")
    print("  y = x * 2")
    print("  z = sin(y)")
    print("  return sum(z ** 2)")
    print(f"\nJaxpr representation:")
    print(jaxpr_complex)
    
    # Example 4: Matrix multiplication - extract HLO
    print("\n" + "-" * 70)
    print("4. HLO/StableHLO EXTRACTION: Matrix Multiplication")
    print("-" * 70)
    A = jnp.ones((3, 4))
    B = jnp.ones((4, 5))
    lowered_matmul = jit(matrix_multiply).lower(A, B)
    hlo_text_matmul = lowered_matmul.as_text()
    print(f"\nFunction: matmul(A, B)")
    print(f"Input shapes: A={A.shape}, B={B.shape}")
    print(f"Output shape: {(3, 5)}")
    print(f"\nStableHLO Text (truncated to first 800 chars):")
    print(hlo_text_matmul[:800])
    if len(hlo_text_matmul) > 800:
        print("...")
    
    # Example 5: Save Python→IR pair
    print("\n" + "-" * 70)
    print("5. SAVING PYTHON→IR PAIRS")
    print("-" * 70)
    
    # Save Jaxpr example
    python_code = '''def simple_add(x, y):
    return x + y'''
    
    jaxpr_str = str(jaxpr)
    
    pair = {
        "python_code": python_code,
        "ir_type": "jaxpr",
        "ir_code": jaxpr_str,
        "input_shapes": [list(x.shape), list(y.shape)],
        "input_dtypes": [str(x.dtype), str(y.dtype)]
    }
    
    print("\nExample Python→Jaxpr pair:")
    print(f"Python code:\n{pair['python_code']}")
    print(f"\nJaxpr:\n{pair['ir_code']}")
    
    print("\n" + "=" * 70)
    print("SUCCESS: IR extraction working for both Jaxpr and HLO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
