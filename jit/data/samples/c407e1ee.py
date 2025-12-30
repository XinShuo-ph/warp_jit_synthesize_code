import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(y)
    v1 = jnp.abs(v0)
    v2 = jnp.add(v0, v0)
    v3 = jnp.add(v2, v0)
    v4 = jnp.multiply(x, v3)
    v5 = jnp.tanh(x)
    v6 = jnp.exp(v1)
    v7 = jnp.square(x)
    return v7