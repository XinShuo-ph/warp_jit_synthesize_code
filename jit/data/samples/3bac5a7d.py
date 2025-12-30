import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.add(v0, v0)
    v2 = jnp.cos(v0)
    v3 = jnp.sin(y)
    v4 = jnp.exp(x)
    v5 = jnp.exp(x)
    v6 = jnp.multiply(v0, v2)
    return v6