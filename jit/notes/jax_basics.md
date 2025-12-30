# JAX Basics

## JIT Compilation Flow
1. Python function → `jax.jit` decorator
2. First call triggers tracing with abstract values
3. Trace produces JAXPR (JAX Program Representation)
4. JAXPR lowered to StableHLO (XLA intermediate)
5. XLA compiles to device-specific code (CPU/GPU/TPU)

## Key APIs for IR Extraction

### JAXPR (High-level IR)
```python
jaxpr = jax.make_jaxpr(fn)(*example_args)
print(jaxpr)  # Human-readable JAXPR
```

### XLA HLO (Low-level IR)
```python
jitted = jax.jit(fn)
lowered = jitted.lower(*example_args)
hlo_text = lowered.as_text()  # StableHLO text
```

## JAXPR Structure
- `lambda ; <inputs>`: Input variables with types (e.g., `a:f32[4]`)
- `let ... in`: Computation body with primitive operations
- Primitives: `add`, `mul`, `sin`, `reduce_sum`, `scan`, etc.
- Final `in (<outputs>,)`: Return values

## Key Transformations
- `jax.jit`: JIT compilation
- `jax.vmap`: Automatic vectorization (batching)
- `jax.grad`: Automatic differentiation
- `jax.lax.scan`: Sequential loops (generates `scan` primitive)

## Types
- `f32[n]`: Float32 array of shape (n,)
- `f32[]`: Scalar float32
- `i32[n,m]`: Int32 array of shape (n, m)
