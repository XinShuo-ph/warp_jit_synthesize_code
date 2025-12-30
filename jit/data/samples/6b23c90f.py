import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.multiply(v0, y)
    v2 = jnp.multiply(v1, y)
    v3 = jnp.sin(v2)
    v4 = jnp.abs(v2)
    v5 = jnp.tanh(v3)
    v6 = jnp.square(v5)
    v7 = jnp.exp(v2)
    v8 = jnp.multiply(v2, v6)
    return v8