import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, y)
    v1 = jnp.sin(y)
    v2 = jnp.cos(x)
    v3 = jnp.tanh(v1)
    v4 = jnp.square(v2)
    return v4