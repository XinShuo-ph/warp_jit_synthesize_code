# JAX Compilation Flow

## Overview

JAX is a high-performance numerical computing library that uses XLA (Accelerated Linear Algebra) for just-in-time compilation. This document explains how JAX transforms Python functions into optimized machine code.

## Compilation Stages

### 1. Python Function → Jaxpr
When you apply `jax.jit` to a function, JAX first traces it to create a Jaxpr (JAX expression), an intermediate representation.

```python
import jax
import jax.numpy as jnp

def f(x):
    return x * 2 + 1

# Get Jaxpr
jaxpr = jax.make_jaxpr(f)(jnp.ones(5))
print(jaxpr)
# { lambda ; a:f32[5]. let b:f32[5] = mul a 2.0; c:f32[5] = add b 1.0 in (c,) }
```

### 2. Jaxpr → HLO (High Level Operations)
The Jaxpr is lowered to HLO, XLA's intermediate representation:

```python
lowered = jax.jit(f).lower(jnp.ones(5))
print(lowered.as_text())  # HLO text representation
```

### 3. HLO → Optimized HLO
XLA applies optimization passes:
- Fusion
- Layout optimization
- Common subexpression elimination
- Dead code elimination

```python
compiled = lowered.compile()
print(compiled.as_text())  # Optimized HLO
```

### 4. Optimized HLO → Machine Code
XLA generates platform-specific code:
- CPU: Uses LLVM backend
- GPU: Uses NVIDIA's CUDA compiler or AMD's ROCm

## Key Transformations

### JIT Compilation
```python
@jax.jit
def fast_fn(x):
    return jnp.sin(x) + jnp.cos(x)
```

### Automatic Differentiation
```python
# Gradient of a scalar function
grad_fn = jax.grad(lambda x: jnp.sum(x ** 2))

# Jacobian for vector functions
jacobian_fn = jax.jacobian(fn)

# Vector-Jacobian product (reverse mode)
vjp_fn = jax.vjp(fn, x)

# Jacobian-Vector product (forward mode)
jvp_fn = jax.jvp(fn, (x,), (v,))
```

### Vectorization (vmap)
```python
# Automatically vectorize over batch dimension
batched_fn = jax.vmap(fn)
```

### Parallelization (pmap)
```python
# Parallelize across multiple devices
parallel_fn = jax.pmap(fn)
```

## HLO Structure

HLO programs consist of:

1. **Computations**: Named function-like units
2. **Instructions**: Operations within computations
3. **Shapes**: Tensor dimensions and types

### Common HLO Operations

| HLO Op | Description |
|--------|-------------|
| `add` | Element-wise addition |
| `multiply` | Element-wise multiplication |
| `dot` | Matrix multiplication |
| `reduce` | Reduction operations (sum, max, etc.) |
| `broadcast` | Broadcasting for shape compatibility |
| `select` | Conditional selection (like `where`) |
| `fusion` | Fused kernel combining multiple ops |

## Example HLO Output

```
HloModule jit_f

ENTRY main.3 {
  p0 = f32[5] parameter(0)
  c1 = f32[] constant(2)
  b1 = f32[5] broadcast(c1), dimensions={}
  m1 = f32[5] multiply(p0, b1)
  c2 = f32[] constant(1)
  b2 = f32[5] broadcast(c2), dimensions={}
  ROOT add = f32[5] add(m1, b2)
}
```

## GPU Execution

On GPU, JAX/XLA:
1. Generates CUDA/HIP kernels
2. Manages memory transfers
3. Handles kernel fusion for efficiency

## Memory Management

JAX uses:
- **Unified memory model**: Single array type for all devices
- **Device buffers**: Data stays on device between operations
- **Async dispatch**: Operations are queued asynchronously
