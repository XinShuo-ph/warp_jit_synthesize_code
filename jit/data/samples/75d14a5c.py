import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, x)
    v1 = jnp.sin(y)
    v2 = jnp.exp(x)
    v3 = jnp.add(x, v0)
    v4 = jnp.square(v2)
    v5 = jnp.multiply(v3, x)
    v6 = jnp.square(v1)
    v7 = jnp.maximum(v1, v5)
    return v7