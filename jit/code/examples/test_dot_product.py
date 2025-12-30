"""JAX dot product example with IR extraction."""
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '/workspace/jit/code/extraction')

from ir_extractor import extract_ir


def dot_product(a, b):
    """Compute dot product of two vectors."""
    return jnp.sum(a * b)


if __name__ == "__main__":
    # Create sample inputs
    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, (1000,))
    b = jax.random.normal(jax.random.PRNGKey(43), (1000,))
    
    # Test the function
    result = dot_product(a, b)
    print(f"Dot product result: {result}")
    
    # Verify against jnp.dot
    expected = jnp.dot(a, b)
    print(f"Expected (jnp.dot): {expected}")
    print(f"Match: {jnp.allclose(result, expected)}")
    
    # Extract IR
    ir = extract_ir(dot_product, (a, b))
    
    print("\n=== Jaxpr ===")
    print(ir.jaxpr_text)
    
    print("\n=== HLO (first 1500 chars) ===")
    print(ir.hlo_text[:1500])
    
    # Test gradient
    grad_fn = jax.grad(dot_product)
    grad_a = grad_fn(a, b)
    print(f"\nGradient w.r.t. a (first 5): {grad_a[:5]}")
    print(f"Expected (b): {b[:5]}")
    print(f"Gradient match: {jnp.allclose(grad_a, b)}")
