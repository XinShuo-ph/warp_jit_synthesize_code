import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, x)
    v1 = jnp.cos(x)
    v2 = jnp.tanh(v0)
    v3 = jnp.add(y, v1)
    v4 = jnp.minimum(x, v1)
    v5 = jnp.add(v4, v1)
    v6 = jnp.cos(v2)
    return v6