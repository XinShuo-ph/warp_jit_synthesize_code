import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(y)
    v1 = jnp.minimum(v0, y)
    v2 = jnp.add(v0, v1)
    v3 = jnp.exp(x)
    v4 = jnp.minimum(y, v3)
    v5 = jnp.multiply(v2, v1)
    return v5