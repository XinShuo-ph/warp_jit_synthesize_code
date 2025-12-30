"""Matrix operations with JAX JIT compilation."""
import jax
import jax.numpy as jnp


@jax.jit
def matmul(A, B):
    """Matrix multiplication."""
    return A @ B


@jax.jit
def matrix_transform(A, x, b):
    """Compute Ax + b (affine transformation)."""
    return A @ x + b


@jax.jit
def batch_normalize(x, eps=1e-5):
    """Simple batch normalization: (x - mean) / std."""
    mean = jnp.mean(x, axis=0)
    std = jnp.std(x, axis=0)
    return (x - mean) / (std + eps)


@jax.jit
def softmax(x):
    """Softmax along last axis."""
    x_max = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


def main():
    key = jax.random.PRNGKey(42)
    
    # Test matmul
    A = jax.random.normal(key, (3, 4))
    B = jax.random.normal(key, (4, 2))
    result1 = matmul(A, B)
    print(f"matmul shape: {result1.shape}")
    
    # Test matrix_transform
    A = jax.random.normal(key, (3, 3))
    x = jax.random.normal(key, (3,))
    b = jax.random.normal(key, (3,))
    result2 = matrix_transform(A, x, b)
    print(f"matrix_transform: {result2}")
    
    # Test batch_normalize
    x = jax.random.normal(key, (8, 4))
    result3 = batch_normalize(x)
    print(f"batch_normalize mean: {jnp.mean(result3, axis=0)}")
    
    # Test softmax
    logits = jax.random.normal(key, (2, 5))
    result4 = softmax(logits)
    print(f"softmax sum: {jnp.sum(result4, axis=-1)}")
    
    return result1, result2, result3, result4


if __name__ == "__main__":
    r1, r2, r3, r4 = main()
    print("\nRun 2:")
    r1_2, r2_2, r3_2, r4_2 = main()
    
    # Verify consistency
    assert jnp.allclose(r1, r1_2)
    assert jnp.allclose(r2, r2_2)
    assert jnp.allclose(r3, r3_2)
    assert jnp.allclose(r4, r4_2)
    print("\nConsistency check passed!")
