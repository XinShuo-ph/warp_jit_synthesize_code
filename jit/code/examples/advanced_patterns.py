"""Advanced JAX patterns: vmap, scan, while_loop, custom gradients."""

import jax
import jax.numpy as jnp
from jax import lax


def example_vmap():
    """Vectorized map over batches."""
    def single_fn(x, y):
        return jnp.dot(x, y)
    
    batched_fn = jax.vmap(single_fn)
    
    x = jnp.ones((8, 4))
    y = jnp.ones((8, 4))
    
    jaxpr = jax.make_jaxpr(batched_fn)(x, y)
    print("=== vmap (batched dot) ===")
    print(f"Input: batch of 8 vectors of dim 4")
    print(f"Jaxpr:\n{jaxpr}")
    return jaxpr


def example_scan():
    """Scan for sequential computation (RNN-like)."""
    def step_fn(carry, x):
        new_carry = carry * 0.9 + x * 0.1
        return new_carry, new_carry
    
    def scan_fn(init, xs):
        return lax.scan(step_fn, init, xs)
    
    init = jnp.zeros(4)
    xs = jnp.ones((10, 4))
    
    jaxpr = jax.make_jaxpr(scan_fn)(init, xs)
    print("\n=== scan (sequential) ===")
    print(f"Input: init [4], sequence [10, 4]")
    print(f"Jaxpr:\n{jaxpr}")
    return jaxpr


def example_while_loop():
    """While loop for iterative algorithms."""
    def cond_fn(state):
        i, _ = state
        return i < 10
    
    def body_fn(state):
        i, x = state
        return i + 1, x * 1.1
    
    def loop_fn(x):
        init_state = (0, x)
        _, result = lax.while_loop(cond_fn, body_fn, init_state)
        return result
    
    x = jnp.ones(4)
    jaxpr = jax.make_jaxpr(loop_fn)(x)
    print("\n=== while_loop (iterative) ===")
    print(f"Input: vector [4], loop 10 times")
    print(f"Jaxpr:\n{jaxpr}")
    return jaxpr


def example_cond():
    """Conditional branching."""
    def cond_fn(x, threshold):
        return lax.cond(
            jnp.mean(x) > threshold,
            lambda: x * 2,
            lambda: x / 2
        )
    
    x = jnp.ones(4)
    jaxpr = jax.make_jaxpr(cond_fn)(x, 0.5)
    print("\n=== cond (branching) ===")
    print(f"Input: vector [4], threshold scalar")
    print(f"Jaxpr:\n{jaxpr}")
    return jaxpr


def example_grad():
    """Automatic differentiation."""
    def loss_fn(params, x, y):
        pred = jnp.dot(x, params)
        return jnp.mean((pred - y) ** 2)
    
    grad_fn = jax.grad(loss_fn)
    
    params = jnp.ones(4)
    x = jnp.ones((8, 4))
    y = jnp.ones(8)
    
    jaxpr = jax.make_jaxpr(grad_fn)(params, x, y)
    print("\n=== grad (autodiff) ===")
    print(f"Input: params [4], x [8,4], y [8]")
    print(f"Jaxpr:\n{jaxpr}")
    return jaxpr


def example_vmap_in_axes():
    """vmap with different in_axes."""
    def fn(weight, batch):
        return jnp.dot(batch, weight)
    
    # vmap over batch dim only, broadcast weight
    batched = jax.vmap(fn, in_axes=(None, 0))
    
    w = jnp.ones(4)
    batch = jnp.ones((8, 4))
    
    jaxpr = jax.make_jaxpr(batched)(w, batch)
    print("\n=== vmap with in_axes ===")
    print(f"Input: weight [4] (broadcast), batch [8, 4]")
    print(f"Jaxpr:\n{jaxpr}")
    return jaxpr


if __name__ == "__main__":
    example_vmap()
    example_scan()
    example_while_loop()
    example_cond()
    example_grad()
    example_vmap_in_axes()
    print("\n=== All advanced patterns completed ===")
