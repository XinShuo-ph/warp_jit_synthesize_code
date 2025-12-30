import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(y)
    v1 = jnp.add(x, x)
    v2 = jnp.square(v0)
    v3 = jnp.multiply(x, y)
    v4 = jnp.add(x, v3)
    return v4