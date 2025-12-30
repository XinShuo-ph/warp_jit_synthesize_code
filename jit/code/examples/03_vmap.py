"""
JAX vectorization examples
Demonstrates jax.vmap for automatic vectorization
"""

import jax
import jax.numpy as jnp


def apply_matrix(A, x):
    """Apply matrix to single vector."""
    return jnp.dot(A, x)


def compute_norm(x):
    """Compute L2 norm of a vector."""
    return jnp.sqrt(jnp.sum(x ** 2))


def pairwise_distance(x, y):
    """Compute Euclidean distance between two vectors."""
    return jnp.sqrt(jnp.sum((x - y) ** 2))


def polynomial(x, coeffs):
    """Evaluate polynomial at x with given coefficients."""
    return jnp.sum(coeffs * (x ** jnp.arange(len(coeffs))))


def main():
    print("=" * 60)
    print("JAX Vectorization (vmap) Examples")
    print("=" * 60)
    
    # Example 1: Batch matrix-vector multiplication
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    vectors = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    
    print("\n1. Batch matrix-vector multiplication")
    print(f"Matrix A:\n{A}")
    print(f"Batch of vectors:\n{vectors}")
    
    # Without vmap (manual loop)
    results_manual = jnp.stack([apply_matrix(A, v) for v in vectors])
    print(f"Results (manual): {results_manual}")
    
    # With vmap (automatic vectorization)
    batched_apply = jax.vmap(apply_matrix, in_axes=(None, 0))
    results_vmap = batched_apply(A, vectors)
    print(f"Results (vmap): {results_vmap}")
    
    # Example 2: Batch norm computation
    print("\n2. Batch norm computation")
    vectors = jnp.array([[3.0, 4.0], [1.0, 0.0], [2.0, 2.0]])
    print(f"Vectors: {vectors}")
    
    norms = jax.vmap(compute_norm)(vectors)
    print(f"Norms: {norms}")
    
    # Example 3: Pairwise distances
    print("\n3. Pairwise distances between vectors")
    x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y_batch = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    
    distances = jax.vmap(pairwise_distance)(x_batch, y_batch)
    print(f"x_batch: {x_batch}")
    print(f"y_batch: {y_batch}")
    print(f"Distances: {distances}")
    
    # Example 4: Batch polynomial evaluation
    print("\n4. Batch polynomial evaluation")
    x_values = jnp.array([0.0, 1.0, 2.0, 3.0])
    coeffs = jnp.array([1.0, 2.0, 3.0])  # 1 + 2x + 3x^2
    
    results = jax.vmap(polynomial, in_axes=(0, None))(x_values, coeffs)
    print(f"Polynomial: 1 + 2x + 3x^2")
    print(f"x values: {x_values}")
    print(f"Results: {results}")
    
    # Example 5: Combining vmap with jit
    print("\n5. Combined vmap + jit")
    @jax.jit
    @jax.vmap
    def fast_squared_distance(x, y):
        diff = x - y
        return jnp.sum(diff * diff)
    
    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    distances = fast_squared_distance(x, y)
    print(f"Squared distances (vmap+jit): {distances}")
    
    print("\n" + "=" * 60)
    print("All vmap examples ran successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
