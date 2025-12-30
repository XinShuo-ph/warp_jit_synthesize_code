import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.add(v0, v0)
    v2 = jnp.square(v0)
    v3 = jnp.multiply(y, v0)
    v4 = jnp.minimum(v3, v0)
    v5 = jnp.maximum(v4, v4)
    v6 = jnp.add(y, v1)
    return v6