import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, x)
    v1 = jnp.add(v0, x)
    v2 = jnp.exp(v0)
    v3 = jnp.tanh(x)
    v4 = jnp.multiply(y, v2)
    v5 = jnp.tanh(v1)
    v6 = jnp.abs(y)
    return v6