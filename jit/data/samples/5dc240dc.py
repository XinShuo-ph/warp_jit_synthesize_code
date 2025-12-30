import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(x)
    v1 = jnp.abs(x)
    v2 = jnp.subtract(v1, y)
    v3 = jnp.cos(v2)
    v4 = jnp.sin(v2)
    v5 = jnp.exp(v4)
    v6 = jnp.tanh(v0)
    v7 = jnp.add(v2, v5)
    v8 = jnp.add(v5, v5)
    return v8