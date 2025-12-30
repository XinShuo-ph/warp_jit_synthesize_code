"""Simple JAX addition example with IR extraction."""
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '/workspace/jit/code/extraction')

from ir_extractor import extract_ir


def add_kernel(a, b):
    """Elementwise addition of two arrays."""
    return a + b


if __name__ == "__main__":
    # Create sample inputs
    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, (100,))
    b = jax.random.normal(jax.random.PRNGKey(43), (100,))
    
    # Test the function
    result = add_kernel(a, b)
    print(f"Result shape: {result.shape}")
    print(f"Result (first 5): {result[:5]}")
    
    # Extract IR
    ir = extract_ir(add_kernel, (a, b))
    
    print("\n=== Jaxpr ===")
    print(ir.jaxpr_text[:1000])
    
    print("\n=== HLO (first 1000 chars) ===")
    print(ir.hlo_text[:1000])
    
    # JIT compile and run
    jit_add = jax.jit(add_kernel)
    result_jit = jit_add(a, b)
    print(f"\nJIT Result matches: {jnp.allclose(result, result_jit)}")
