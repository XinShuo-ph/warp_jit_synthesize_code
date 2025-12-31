# JAX JIT Compilation Flow

## Compilation Pipeline
1. **Python function** → decorated with `@jax.jit`
2. **Tracing** → JAX traces function with abstract values to build JAXPR
3. **JAXPR** → High-level primitive representation (jax.make_jaxpr)
4. **StableHLO** → MLIR-based IR (lowered.as_text())
5. **XLA HLO** → XLA's computation graph (compiler_ir().as_hlo_text())
6. **Compilation** → XLA compiles to optimized machine code

## IR Extraction APIs
```python
# JAXPR (JAX primitives)
jax.make_jaxpr(fn)(*args)

# StableHLO (MLIR format)
jax.jit(fn).lower(*args).as_text()

# XLA HLO (classic HLO format)
jax.jit(fn).lower(*args).compiler_ir(dialect='hlo').as_hlo_text()
```

## Key Observations
- JAXPR: Most abstract, closest to Python semantics
- StableHLO: MLIR dialect, includes type info, portable
- XLA HLO: Lowest level, shows actual ops (broadcast, reshape, dot)
- All three are deterministic for same input shapes
- Function must be traceable (no Python control flow on traced values)

## Useful APIs
- `lowered.compile()` → Get compiled executable
- `lowered.cost_analysis()` → Get FLOPs estimate
- `jax.debug.print()` → Debug inside JIT functions
