import jax
import jax.numpy as jnp
from jax import lax

def elementwise_kernel(x, y):
    """Element-wise computation (ax + y)."""
    return x * 2.0 + y

def loop_kernel(x):
    """Loop-based computation (sum reduction manually)."""
    def body_fun(i, val):
        return val + x[i]
    
    return lax.fori_loop(0, x.shape[0], body_fun, 0.0)

def scan_kernel(x):
    """Scan-based computation (cumulative sum)."""
    def body_fun(carry, x_el):
        new_carry = carry + x_el
        return new_carry, new_carry
    
    final, stack = lax.scan(body_fun, 0.0, x)
    return stack

def extract_and_save(func, name, *args):
    print(f"--- Extracting IR for {name} ---")
    lowered = jax.jit(func).lower(*args)
    hlo = lowered.as_text()
    with open(f"jit/data/{name}.hlo", "w") as f:
        f.write(hlo)
    print(f"Saved to jit/data/{name}.hlo")

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (100,))
    y = jax.random.normal(key, (100,))
    
    extract_and_save(elementwise_kernel, "elementwise", x, y)
    extract_and_save(loop_kernel, "loop", x)
    extract_and_save(scan_kernel, "scan", x)
