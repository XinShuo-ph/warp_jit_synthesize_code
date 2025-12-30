import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.add(y, v0)
    v2 = jnp.square(v0)
    v3 = jnp.subtract(v1, v1)
    v4 = jnp.exp(v2)
    v5 = jnp.add(v3, v3)
    v6 = jnp.sin(v5)
    v7 = jnp.sin(v5)
    return v7