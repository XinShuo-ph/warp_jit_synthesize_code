"""JAX transformations: vmap, grad, scan."""
import jax
import jax.numpy as jnp


def square(x):
    """Square a number."""
    return x ** 2


def loss_fn(params, x, y):
    """Simple MSE loss."""
    pred = params @ x
    return jnp.mean((pred - y) ** 2)


def cumsum_scan(xs):
    """Cumulative sum using scan."""
    def step(carry, x):
        new_carry = carry + x
        return new_carry, new_carry
    _, cumsum = jax.lax.scan(step, 0.0, xs)
    return cumsum


if __name__ == "__main__":
    # vmap: vectorize over batch dimension
    print("=== vmap ===")
    xs = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batched_sum = jax.vmap(jnp.sum)
    print(f"batched_sum: {batched_sum(xs)}")
    print("JAXPR:", jax.make_jaxpr(batched_sum)(xs))
    
    # grad: automatic differentiation
    print("\n=== grad ===")
    grad_square = jax.grad(square)
    print(f"d/dx(x^2) at x=3: {grad_square(3.0)}")
    print("JAXPR:", jax.make_jaxpr(grad_square)(3.0))
    
    # scan: sequential processing
    print("\n=== scan ===")
    arr = jnp.array([1.0, 2.0, 3.0, 4.0])
    print(f"cumsum: {cumsum_scan(arr)}")
    print("JAXPR:", jax.make_jaxpr(cumsum_scan)(arr))
