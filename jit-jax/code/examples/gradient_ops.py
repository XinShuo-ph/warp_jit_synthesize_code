"""Gradient computation with JAX."""
import jax
import jax.numpy as jnp


def loss_fn(w, x, y):
    """Mean squared error loss."""
    pred = jnp.dot(x, w)
    return jnp.mean((pred - y) ** 2)


# Create gradient function
grad_loss = jax.jit(jax.grad(loss_fn))


def quadratic(x):
    """f(x) = x^2, gradient should be 2x."""
    return x ** 2


grad_quadratic = jax.jit(jax.grad(quadratic))


def rosenbrock(xy):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2."""
    x, y = xy[0], xy[1]
    return (1 - x)**2 + 100 * (y - x**2)**2


grad_rosenbrock = jax.jit(jax.grad(rosenbrock))


def multi_output(x):
    """Function returning sum of outputs for gradient."""
    return jnp.sum(jnp.sin(x) + x**2)


grad_multi = jax.jit(jax.grad(multi_output))


def main():
    key = jax.random.PRNGKey(42)
    
    # Test grad_loss
    w = jax.random.normal(key, (4,))
    x = jax.random.normal(key, (8, 4))
    y = jax.random.normal(key, (8,))
    g1 = grad_loss(w, x, y)
    print(f"grad_loss shape: {g1.shape}")
    
    # Test grad_quadratic (should be 2x)
    x = 3.0
    g2 = grad_quadratic(x)
    print(f"grad_quadratic at x=3: {g2} (expected 6.0)")
    assert jnp.isclose(g2, 6.0)
    
    # Test grad_rosenbrock
    xy = jnp.array([0.0, 0.0])
    g3 = grad_rosenbrock(xy)
    print(f"grad_rosenbrock at (0,0): {g3}")
    
    # Test grad_multi
    x = jnp.array([0.0, 1.0, 2.0])
    g4 = grad_multi(x)
    print(f"grad_multi: {g4}")
    
    return g1, g2, g3, g4


if __name__ == "__main__":
    r1, r2, r3, r4 = main()
    print("\nRun 2:")
    r1_2, r2_2, r3_2, r4_2 = main()
    
    # Verify consistency
    assert jnp.allclose(r1, r1_2)
    assert jnp.isclose(r2, r2_2)
    assert jnp.allclose(r3, r3_2)
    assert jnp.allclose(r4, r4_2)
    print("\nConsistency check passed!")
