import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.sin(v0)
    v2 = jnp.square(v1)
    v3 = jnp.tanh(v2)
    v4 = jnp.add(v1, v0)
    v5 = jnp.multiply(v3, v1)
    return v5