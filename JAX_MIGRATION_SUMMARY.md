# JAX Migration Summary

## What Was Done

Adapted the original Warp-based JIT code synthesis instructions to use Google JAX instead.

## Files Created

1. **`instructions_jax.md`** - Complete instructions for JAX-based IR extraction
   - 5 milestones (M1-M5) adapted for JAX
   - JAX-specific technical details
   - IR extraction methods (Jaxpr, HLO, StableHLO)
   - Common pitfalls and solutions
   - Sample code templates

2. **`WARP_TO_JAX_MIGRATION.md`** - Migration guide
   - Side-by-side comparison: Warp vs JAX
   - Concept mapping for each feature
   - IR format examples
   - Migration strategy
   - Dataset quality considerations

3. **`STATE.md`** - Updated project state
   - Set to M1 (JAX Edition)
   - Ready to begin implementation

## Key Differences: Warp → JAX

| Aspect | Warp | JAX |
|--------|------|-----|
| **Use case** | Physics simulation | ML research |
| **IR output** | PTX assembly | Jaxpr, XLA HLO, StableHLO |
| **Paradigm** | Imperative kernels | Functional transformations |
| **Gradients** | wp.Tape() | jax.grad() |
| **Vectorization** | wp.launch() | jax.vmap() |

## Why JAX is Better for LLM Training Data

1. **Multiple IR levels**: Get both high-level (Jaxpr) and low-level (HLO) representations
2. **Richer transformations**: grad, vmap, pmap, scan, cond, while_loop
3. **More ML-relevant**: Directly applicable to ML compiler optimization
4. **Easier generation**: No explicit GPU kernel writing needed
5. **Better documentation**: Extensive examples and community support

## Milestone Roadmap

- **M1**: Install JAX, extract basic Jaxpr/HLO from simple functions
- **M2**: Build IR extractor supporting multiple formats
- **M3**: Extract IR from transformed functions (grad, vmap, scan)
- **M4**: Automated synthesis pipeline generating 10+ function categories
- **M5**: Scale to 10k+ Python→IR pairs

## Next Steps

1. Install JAX: `pip install jax jaxlib`
2. Create directory structure: `jit/code/`, `jit/data/`, `jit/notes/`, `jit/tasks/`
3. Start M1: Run basic JAX examples, understand IR extraction
4. Follow `instructions_jax.md` for detailed implementation plan

## Technical Highlights

### IR Extraction (Core Method)
```python
import jax
import jax.numpy as jnp

def my_function(x):
    return jnp.sin(x) + x * 2

# Extract Jaxpr (readable)
jaxpr = jax.make_jaxpr(my_function)(jnp.array(1.0))

# Extract XLA HLO (detailed)
computation = jax.xla_computation(my_function)(jnp.array(1.0))
hlo_text = computation.as_hlo_text()
```

### Function Categories to Generate
1. Arithmetic operations
2. Array operations  
3. Math functions
4. Linear algebra
5. Reductions
6. Indexing
7. **Gradients** (unique to JAX)
8. **Vectorization** (vmap)
9. **Conditionals** (lax.cond)
10. **Loops** (lax.scan, while_loop)

Categories 7-10 are much richer in JAX than Warp's kernel model.

## Estimated Dataset Quality

- **10k+ samples** across 10 categories
- **Multiple IR formats** per sample (Jaxpr + HLO)
- **Transformation variants** (forward pass, gradient, vectorized)
- **Shape/dtype diversity** (scalars to tensors, multiple dtypes)

Expected dataset size: **20k+ unique Python→IR pairs** (including transformation variants)

---

Ready to begin implementation following `instructions_jax.md`!
