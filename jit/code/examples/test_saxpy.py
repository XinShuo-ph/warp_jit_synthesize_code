"""JAX SAXPY (scalar * x + y) example with IR extraction."""
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '/workspace/jit/code/extraction')

from ir_extractor import extract_ir


def saxpy(alpha, x, y):
    """Compute alpha * x + y (SAXPY operation)."""
    return alpha * x + y


if __name__ == "__main__":
    # Create sample inputs
    key = jax.random.PRNGKey(42)
    alpha = 2.5
    x = jax.random.normal(key, (1000,))
    y = jax.random.normal(jax.random.PRNGKey(43), (1000,))
    
    # Test the function
    result = saxpy(alpha, x, y)
    print(f"Result shape: {result.shape}")
    print(f"Result (first 5): {result[:5]}")
    
    # Verify manually
    expected = alpha * x + y
    print(f"Match: {jnp.allclose(result, expected)}")
    
    # Extract IR
    ir = extract_ir(saxpy, (alpha, x, y))
    
    print("\n=== Jaxpr ===")
    print(ir.jaxpr_text)
    
    print("\n=== HLO (first 1500 chars) ===")
    print(ir.hlo_text[:1500])
    
    # Test gradient w.r.t. first array argument
    def loss_fn(x):
        return jnp.sum(saxpy(alpha, x, y))
    
    grad_fn = jax.grad(loss_fn)
    grad_x = grad_fn(x)
    print(f"\nGradient w.r.t. x (first 5): {grad_x[:5]}")
    print(f"Expected (alpha): {alpha}")
    print(f"All gradients equal alpha: {jnp.allclose(grad_x, alpha)}")
