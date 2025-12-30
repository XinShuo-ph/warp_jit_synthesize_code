# JAX JIT Compilation Flow

## Compilation Pipeline
Python function → JAXPR → StableHLO → XLA HLO → Machine code

## IR Extraction Methods

### 1. JAXPR (JAX Program Representation)
```python
jaxpr = jax.make_jaxpr(fn)(*args)
```
- High-level, Python-like representation
- Shows operations as equations: `c:f32[2] = sin a`
- Good for understanding JAX transformations

### 2. StableHLO (pre-compilation)
```python
lowered = jax.jit(fn).lower(*args)
hlo_text = lowered.as_text()
```
- MLIR-based IR dialect
- Operations like `stablehlo.sine`, `stablehlo.add`
- Portable across different XLA backends

### 3. Compiled HLO (post-optimization)
```python
compiled = lowered.compile()
optimized_hlo = compiled.as_text()
```
- XLA-optimized IR with fusions
- Contains metadata (op_names, stack frames)
- Backend-specific optimizations applied

## Key APIs
- `jax.jit(fn)` - JIT compilation decorator
- `jax.jit(fn).lower(*args)` - Get lowered representation
- `jax.make_jaxpr(fn)(*args)` - Get JAXPR
- `jax.grad(fn)` - Automatic differentiation
- `jax.vmap(fn)` - Vectorization transformation

## IR Format Notes
- StableHLO uses MLIR syntax: `%0 = stablehlo.op %arg0 : type`
- Tensors typed as `tensor<shape x dtype>` e.g., `tensor<2xf32>`
- Compiled HLO includes fusion information for optimization
