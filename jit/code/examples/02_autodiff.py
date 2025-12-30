"""
JAX automatic differentiation examples
Demonstrates jax.grad for computing derivatives
"""

import jax
import jax.numpy as jnp


def f(x):
    """Simple quadratic function: f(x) = x^2 + 2x + 1"""
    return x ** 2 + 2 * x + 1


def g(x):
    """Sum of exponentials: g(x) = sum(exp(x))"""
    return jnp.sum(jnp.exp(x))


def h(x, y):
    """Two-argument function: h(x, y) = x^2 * y + sin(x * y)"""
    return x ** 2 * y + jnp.sin(x * y)


def loss_fn(params, x, y):
    """Simple loss function for demonstration."""
    pred = params[0] * x + params[1]
    return jnp.mean((pred - y) ** 2)


def main():
    print("=" * 60)
    print("JAX Automatic Differentiation Examples")
    print("=" * 60)
    
    # Compute gradient of f
    grad_f = jax.grad(f)
    x = 3.0
    print(f"\nf(x) = x^2 + 2x + 1")
    print(f"f({x}) = {f(x)}")
    print(f"f'({x}) = {grad_f(x)} (expected: 2*3 + 2 = 8)")
    
    # Compute gradient of g
    grad_g = jax.grad(g)
    x = jnp.array([1.0, 2.0, 3.0])
    print(f"\ng(x) = sum(exp(x))")
    print(f"g({x}) = {g(x)}")
    print(f"∇g({x}) = {grad_g(x)} (expected: exp(x))")
    
    # Compute partial derivatives of h
    dh_dx = jax.grad(h, argnums=0)
    dh_dy = jax.grad(h, argnums=1)
    x, y = 2.0, 3.0
    print(f"\nh(x, y) = x^2 * y + sin(x * y)")
    print(f"h({x}, {y}) = {h(x, y)}")
    print(f"∂h/∂x({x}, {y}) = {dh_dx(x, y)}")
    print(f"∂h/∂y({x}, {y}) = {dh_dy(x, y)}")
    
    # Compute gradient of loss function
    grad_loss = jax.grad(loss_fn, argnums=0)
    params = jnp.array([2.0, 1.0])
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([3.0, 5.0, 7.0])
    print(f"\nloss_fn gradient:")
    print(f"params = {params}, x = {x}, y = {y}")
    print(f"loss = {loss_fn(params, x, y)}")
    print(f"∇loss = {grad_loss(params, x, y)}")
    
    print("\n" + "=" * 60)
    print("All autodiff examples ran successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
