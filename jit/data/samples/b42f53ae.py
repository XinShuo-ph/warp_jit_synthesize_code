import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(y)
    v1 = jnp.multiply(x, x)
    v2 = jnp.square(v1)
    v3 = jnp.add(v0, y)
    v4 = jnp.multiply(v1, v2)
    v5 = jnp.minimum(x, x)
    v6 = jnp.cos(v5)
    return v6