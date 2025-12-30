import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.multiply(v0, y)
    v2 = jnp.cos(v0)
    v3 = jnp.square(v0)
    v4 = jnp.minimum(v1, v2)
    v5 = jnp.subtract(y, y)
    v6 = jnp.add(x, v0)
    return v6