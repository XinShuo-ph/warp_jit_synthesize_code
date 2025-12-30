import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, y)
    v1 = jnp.multiply(x, x)
    v2 = jnp.subtract(v0, x)
    v3 = jnp.add(x, v1)
    v4 = jnp.tanh(v2)
    v5 = jnp.multiply(x, v4)
    return v5