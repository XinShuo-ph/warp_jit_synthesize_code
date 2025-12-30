# JAX Quick Reference

## Installation

```bash
# CPU-only (fastest to install)
pip install jax jaxlib

# With CUDA 12 support (if GPU available)
pip install -U "jax[cuda12]"

# With CUDA 11 support
pip install -U "jax[cuda11_local]"
```

## Basic IR Extraction

### Extract Jaxpr (High-level, Readable)
```python
import jax
import jax.numpy as jnp

def my_function(x, y):
    return jnp.sin(x) + y * 2

# Create example inputs
x = jnp.array(1.0)
y = jnp.array(2.0)

# Extract Jaxpr
jaxpr = jax.make_jaxpr(my_function)(x, y)
print(jaxpr)
# Output: { lambda ; a:f32[] b:f32[]. let c:f32[] = sin a ... }
```

### Extract XLA HLO (Low-level, Detailed)
```python
# Extract HLO
computation = jax.xla_computation(my_function)(x, y)
hlo_text = computation.as_hlo_text()
print(hlo_text)
# Output: HloModule with detailed operations
```

### Extract Both Formats
```python
def extract_all_ir(fn, *args):
    """Extract all IR formats from a function."""
    return {
        'jaxpr': str(jax.make_jaxpr(fn)(*args)),
        'hlo': jax.xla_computation(fn)(*args).as_hlo_text(),
    }

ir = extract_all_ir(my_function, x, y)
print("Jaxpr:", ir['jaxpr'])
print("HLO:", ir['hlo'])
```

## Common Patterns

### 1. Basic Math Operations
```python
def arithmetic(x, y):
    return x + y * 2 - x / y

def math_ops(x):
    return jnp.sin(x) * jnp.exp(jnp.cos(x))

def array_ops(x):
    return jnp.sum(x ** 2, axis=-1)
```

### 2. Gradient Computation
```python
# Gradient of scalar function
def loss_fn(x):
    return jnp.sum(x ** 2)

grad_fn = jax.grad(loss_fn)
gradient = grad_fn(x)

# Value and gradient together
value_and_grad_fn = jax.value_and_grad(loss_fn)
loss, grad = value_and_grad_fn(x)

# Extract IR of gradient function
jaxpr = jax.make_jaxpr(grad_fn)(x)
```

### 3. Vectorization (vmap)
```python
# Scalar function
def scalar_fn(x):
    return x ** 2 + jnp.sin(x)

# Vectorize it
vector_fn = jax.vmap(scalar_fn)

# Apply to array
result = vector_fn(jnp.array([1.0, 2.0, 3.0]))

# Extract IR of vectorized version
jaxpr = jax.make_jaxpr(vector_fn)(jnp.array([1.0, 2.0, 3.0]))
```

### 4. Conditionals
```python
# Use jax.lax.cond (not Python if!)
def conditional_fn(x):
    return jax.lax.cond(
        x > 0,
        lambda x: x * 2,      # true branch
        lambda x: x / 2,      # false branch
        x
    )

# Or use jax.lax.select
def select_fn(x, y, z):
    return jax.lax.select(x > 0, y, z)
```

### 5. Loops
```python
# Use jax.lax.scan (efficient loop)
def scan_fn(init, n):
    def body(carry, i):
        carry = carry + i
        return carry, carry
    
    final, history = jax.lax.scan(body, init, jnp.arange(n))
    return final

# Use jax.lax.fori_loop (simple for loop)
def fori_fn(n):
    def body(i, val):
        return val + i
    
    return jax.lax.fori_loop(0, n, body, 0)

# Use jax.lax.while_loop
def while_fn(x):
    def cond(val):
        return val < 100
    
    def body(val):
        return val * 2
    
    return jax.lax.while_loop(cond, body, x)
```

### 6. Matrix Operations
```python
def matmul_fn(A, B):
    return jnp.matmul(A, B)

def linear_algebra(A):
    return jnp.linalg.norm(A, axis=-1)

def batch_matmul(A, B):
    return jnp.einsum('bij,bjk->bik', A, B)
```

## JIT Compilation

```python
# Add @jax.jit decorator for compilation
@jax.jit
def compiled_fn(x):
    return jnp.sin(x) + x ** 2

# First call: compiles
result = compiled_fn(jnp.array(1.0))  # Slow

# Subsequent calls: uses cached compilation
result = compiled_fn(jnp.array(2.0))  # Fast
```

## Common Gotchas

### ❌ Don't use Python control flow in JIT
```python
@jax.jit
def bad(x):
    if x > 0:  # Error! Python bool() of traced value
        return x * 2
    return x
```

### ✅ Use jax.lax instead
```python
@jax.jit
def good(x):
    return jax.lax.cond(x > 0, lambda x: x * 2, lambda x: x, x)
```

### ❌ Don't mutate arrays
```python
def bad(x):
    x[0] = 5  # Error! Arrays are immutable
    return x
```

### ✅ Use functional updates
```python
def good(x):
    return x.at[0].set(5)
```

### ❌ Don't use dynamic shapes in JIT
```python
@jax.jit
def bad(x):
    return x[:len(x)//2]  # Error! Dynamic slicing
```

### ✅ Use static shapes or dynamic_slice
```python
@jax.jit
def good(x, n):
    return jax.lax.dynamic_slice(x, (0,), (n,))
```

## Testing & Validation

```python
import jax
import jax.numpy as jnp

# Check if function compiles
try:
    jaxpr = jax.make_jaxpr(my_function)(x, y)
    print("✓ Function compiles successfully")
    print(jaxpr)
except Exception as e:
    print("✗ Compilation failed:", e)

# Check if HLO extraction works
try:
    computation = jax.xla_computation(my_function)(x, y)
    hlo = computation.as_hlo_text()
    print("✓ HLO extraction successful")
    print(f"HLO length: {len(hlo)} characters")
except Exception as e:
    print("✗ HLO extraction failed:", e)

# Verify determinism (run twice, compare)
result1 = my_function(x, y)
result2 = my_function(x, y)
assert jnp.allclose(result1, result2), "Non-deterministic!"
print("✓ Function is deterministic")
```

## Pipeline Template

```python
import jax
import jax.numpy as jnp
import json
import inspect
from typing import Callable, Dict, Any

def generate_training_pair(fn: Callable, *args) -> Dict[str, Any]:
    """Generate a Python→IR training pair."""
    
    # Extract source code
    source_code = inspect.getsource(fn)
    
    # Extract Jaxpr
    jaxpr = str(jax.make_jaxpr(fn)(*args))
    
    # Extract HLO
    computation = jax.xla_computation(fn)(*args)
    hlo = computation.as_hlo_text()
    
    # Test function (verify it runs)
    try:
        result = fn(*args)
        success = True
    except Exception as e:
        result = None
        success = False
    
    return {
        'source_code': source_code,
        'jaxpr': jaxpr,
        'hlo': hlo,
        'test_success': success,
        'input_shapes': [arg.shape for arg in args],
        'input_dtypes': [str(arg.dtype) for arg in args],
    }

# Example usage
def example_fn(x):
    return jnp.sin(x) * 2

pair = generate_training_pair(example_fn, jnp.array(1.0))
print(json.dumps(pair, indent=2))
```

## Useful Commands

```bash
# Check JAX installation
python -c "import jax; print(jax.__version__)"

# Check available devices
python -c "import jax; print(jax.devices())"

# Check if GPU is available
python -c "import jax; print('GPU' if jax.devices()[0].platform == 'gpu' else 'CPU only')"

# Run a quick test
python -c "import jax.numpy as jnp; x = jnp.array([1,2,3]); print(x.sum())"
```

## Resources

- **Official docs**: https://jax.readthedocs.io/
- **GitHub**: https://github.com/google/jax
- **Examples**: https://github.com/google/jax/tree/main/examples
- **Autodiff cookbook**: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
- **Common gotchas**: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
