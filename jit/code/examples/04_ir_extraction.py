"""
JAX IR Extraction Example
Demonstrates how to extract HLO and StableHLO intermediate representations
"""

import jax
import jax.numpy as jnp


def simple_add(x, y):
    """Simple addition function."""
    return x + y


def complex_computation(x):
    """More complex computation with multiple operations."""
    return jnp.tanh(jnp.dot(x, x.T) + 1.0)


def matrix_multiply(A, B):
    """Matrix multiplication."""
    return jnp.dot(A, B)


def conditional_function(x):
    """Function with conditional logic."""
    return jnp.where(x > 0, x ** 2, -x)


def loop_function(x, n):
    """Function with a loop using lax.fori_loop."""
    from jax import lax
    
    def body_fn(i, val):
        return val + x[i]
    
    return lax.fori_loop(0, n, body_fn, 0.0)


def extract_hlo_text(func, *args):
    """Extract HLO text representation from a JAX function."""
    # Use jax.jit().lower() to get HLO
    lowered = jax.jit(func).lower(*args)
    hlo_ir = lowered.compiler_ir(dialect='hlo')
    # Use as_text() to get string representation
    if hasattr(hlo_ir, 'as_hlo_text'):
        return hlo_ir.as_hlo_text()
    return lowered.as_text(dialect='hlo')


def extract_stablehlo_text(func, *args):
    """Extract StableHLO text representation from a JAX function."""
    lowered = jax.jit(func).lower(*args)
    try:
        stablehlo_ir = lowered.compiler_ir(dialect='stablehlo')
        return str(stablehlo_ir)
    except:
        return None


def extract_mhlo_text(func, *args):
    """Extract MHLO text representation from a JAX function."""
    lowered = jax.jit(func).lower(*args)
    try:
        mhlo_ir = lowered.compiler_ir(dialect='mhlo')
        return str(mhlo_ir)
    except:
        return None


def main():
    print("=" * 80)
    print("JAX IR Extraction Examples")
    print("=" * 80)
    
    # Example 1: Simple addition
    print("\n" + "=" * 80)
    print("1. Simple Addition: x + y")
    print("=" * 80)
    
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    
    hlo = extract_hlo_text(simple_add, x, y)
    print("\nHLO Text:")
    print(hlo)
    
    # Example 2: Complex computation
    print("\n" + "=" * 80)
    print("2. Complex Computation: tanh(dot(x, x.T) + 1.0)")
    print("=" * 80)
    
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    
    hlo = extract_hlo_text(complex_computation, x)
    print("\nHLO Text:")
    print(hlo)
    
    # Example 3: Matrix multiplication
    print("\n" + "=" * 80)
    print("3. Matrix Multiplication: dot(A, B)")
    print("=" * 80)
    
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    
    hlo = extract_hlo_text(matrix_multiply, A, B)
    print("\nHLO Text:")
    print(hlo)
    
    # Example 4: Conditional function
    print("\n" + "=" * 80)
    print("4. Conditional Function: where(x > 0, x**2, -x)")
    print("=" * 80)
    
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    hlo = extract_hlo_text(conditional_function, x)
    print("\nHLO Text:")
    print(hlo)
    
    # Example 5: Using jit and checking lowered representation
    print("\n" + "=" * 80)
    print("5. Using jax.jit and lowering")
    print("=" * 80)
    
    @jax.jit
    def jitted_add(x, y):
        return x + y
    
    # Lower the function to get intermediate representation
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    
    lowered = jax.jit(simple_add).lower(x, y)
    print("\nLowered representation available methods:")
    print([m for m in dir(lowered) if not m.startswith('_')])
    
    # Get the HLO from lowered
    hlo_text = lowered.compiler_ir(dialect='hlo')
    print("\nHLO from lowered (first 500 chars):")
    print(str(hlo_text)[:500])
    
    # Try StableHLO if available
    try:
        stablehlo_text = lowered.compiler_ir(dialect='stablehlo')
        print("\nStableHLO from lowered (first 500 chars):")
        print(str(stablehlo_text)[:500])
    except Exception as e:
        print(f"\nStableHLO not available: {e}")
    
    # Example 6: StableHLO
    print("\n" + "=" * 80)
    print("6. StableHLO representation")
    print("=" * 80)
    
    stablehlo = extract_stablehlo_text(simple_add, x, y)
    if stablehlo:
        print("\nStableHLO Text (first 500 chars):")
        print(stablehlo[:500])
    else:
        print("\nStableHLO not available in this JAX version")
    
    # Example 7: MHLO
    print("\n" + "=" * 80)
    print("7. MHLO representation")
    print("=" * 80)
    
    mhlo = extract_mhlo_text(simple_add, x, y)
    if mhlo:
        print("\nMHLO Text (first 500 chars):")
        print(mhlo[:500])
    else:
        print("\nMHLO not available in this JAX version")
    
    print("\n" + "=" * 80)
    print("IR Extraction Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
