import jax
import jax.numpy as jnp

def my_func(x, y):
    return jnp.sin(x) * y + 2.0

x = jnp.array(1.5)
y = jnp.array(2.0)

print("--- JAXPR (Intermediate Representation) ---")
jaxpr = jax.make_jaxpr(my_func)(x, y)
print(jaxpr)

print("\n--- Lowering to HLO (High Level Optimizer) ---")
# Lower the function for specific input shapes/types
lowered = jax.jit(my_func).lower(x, y)
print("HLO text:")
print(lowered.as_text())

print("\n--- Compiled Object ---")
compiled = lowered.compile()
print(f"Compiled executable: {compiled}")

# Execute
result = compiled(x, y)
print(f"Result: {result}")
