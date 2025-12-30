import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(y)
    v1 = jnp.minimum(y, x)
    v2 = jnp.cos(y)
    v3 = jnp.exp(x)
    v4 = jnp.tanh(v3)
    v5 = jnp.multiply(v4, v1)
    v6 = jnp.subtract(v3, v4)
    v7 = jnp.minimum(v5, v1)
    v8 = jnp.exp(v2)
    return v8