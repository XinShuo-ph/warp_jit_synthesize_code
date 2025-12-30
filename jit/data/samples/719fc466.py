import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, x)
    v1 = jnp.add(x, x)
    v2 = jnp.cos(v0)
    v3 = jnp.sin(y)
    v4 = jnp.exp(v2)
    v5 = jnp.square(v1)
    return v5