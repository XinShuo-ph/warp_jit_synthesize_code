"""Explore JAX JIT compilation and IR extraction."""
import jax
import jax.numpy as jnp
from jax import jit

# Example 1: Simple arithmetic function
@jit
def add_vectors(x, y):
    """Add two vectors."""
    return x + y

# Example 2: More complex function
@jit
def saxpy(alpha, x, y):
    """Compute alpha * x + y."""
    return alpha * x + y

# Example 3: Conditional
@jit
def conditional_scale(x, threshold=0.0):
    """Scale values based on threshold."""
    return jnp.where(x > threshold, x * 2.0, x * 0.5)

# Example 4: Reduction
@jit
def sum_squares(x):
    """Compute sum of squares."""
    return jnp.sum(x * x)

def explore_jax_compilation():
    """Explore JAX compilation and IR extraction methods."""
    
    # Create test inputs
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = jnp.array([5.0, 6.0, 7.0, 8.0])
    alpha = 2.5
    
    print("=" * 80)
    print("JAX IR EXPLORATION")
    print("=" * 80)
    
    # Test 1: Get lowered HLO
    print("\n1. LOWERED HLO (add_vectors)")
    print("-" * 80)
    lowered = jax.jit(add_vectors).lower(x, y)
    print(f"Lowered type: {type(lowered)}")
    print(f"Available methods: {[m for m in dir(lowered) if not m.startswith('_')]}")
    
    # Get HLO text
    try:
        hlo_text = lowered.as_text()
        print(f"\nHLO Text (first 500 chars):\n{hlo_text[:500]}")
    except Exception as e:
        print(f"Error getting HLO text: {e}")
    
    # Get compilation trace
    try:
        compiler_ir = lowered.compiler_ir()
        print(f"\nCompiler IR type: {type(compiler_ir)}")
        print(f"Compiler IR (first 500 chars):\n{str(compiler_ir)[:500]}")
    except Exception as e:
        print(f"Error getting compiler IR: {e}")
    
    # Test 2: StableHLO
    print("\n\n2. STABLEHLO (saxpy)")
    print("-" * 80)
    lowered2 = jax.jit(saxpy).lower(alpha, x, y)
    try:
        stablehlo = lowered2.compiler_ir(dialect='stablehlo')
        print(f"StableHLO type: {type(stablehlo)}")
        print(f"StableHLO (first 800 chars):\n{str(stablehlo)[:800]}")
    except Exception as e:
        print(f"Error getting StableHLO: {e}")
    
    # Test 3: MHLO
    print("\n\n3. MHLO (conditional_scale)")
    print("-" * 80)
    lowered3 = jax.jit(conditional_scale).lower(x)
    try:
        mhlo = lowered3.compiler_ir(dialect='mhlo')
        print(f"MHLO type: {type(mhlo)}")
        print(f"MHLO (first 800 chars):\n{str(mhlo)[:800]}")
    except Exception as e:
        print(f"Error getting MHLO: {e}")
    
    # Test 4: Get compiled executable
    print("\n\n4. COMPILED EXECUTABLE")
    print("-" * 80)
    compiled = jax.jit(sum_squares).lower(x).compile()
    print(f"Compiled type: {type(compiled)}")
    print(f"Available methods: {[m for m in dir(compiled) if not m.startswith('_')]}")
    
    # Test 5: XLA computation
    print("\n\n5. XLA COMPUTATION INFO")
    print("-" * 80)
    lowered5 = jax.jit(add_vectors).lower(x, y)
    try:
        # Get cost analysis if available
        if hasattr(lowered5, 'cost_analysis'):
            cost = lowered5.cost_analysis()
            print(f"Cost analysis: {cost}")
    except Exception as e:
        print(f"No cost analysis: {e}")
    
    # Test actual execution
    print("\n\n6. EXECUTION TEST")
    print("-" * 80)
    result = add_vectors(x, y)
    print(f"add_vectors result: {result}")
    print(f"Result type: {type(result)}")
    
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    explore_jax_compilation()
