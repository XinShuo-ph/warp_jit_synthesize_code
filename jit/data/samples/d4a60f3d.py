import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.multiply(x, v0)
    v2 = jnp.minimum(y, x)
    v3 = jnp.exp(v1)
    v4 = jnp.minimum(y, v1)
    v5 = jnp.minimum(v3, v0)
    v6 = jnp.cos(v5)
    v7 = jnp.multiply(y, v3)
    v8 = jnp.minimum(v3, x)
    v9 = jnp.abs(v4)
    return v9