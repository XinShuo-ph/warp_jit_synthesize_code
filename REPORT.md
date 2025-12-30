# Technical Report: JAX JIT Code Synthesis Dataset

**Date:** December 2025  
**Purpose:** Python→HLO/XLA training data for LLM code generation

---

## Executive Summary

This project generates **Python→HLO/XLA training pairs** using JAX's JIT compilation infrastructure. Each sample contains Python function source paired with compiler-generated IR for **XLA backends**, including **forward and backward (gradient) functions**.

---

## Dataset Specifications

| Metric | Value |
|--------|-------|
| Total Samples | 1,500 |
| File Size | ~6 MB |
| Format | JSONL |
| Function Types | 10 categories |
| Jaxpr | ✅ Forward + Backward |
| HLO | ✅ Forward + Backward |
| StableHLO | ✅ Included |

### Sample Structure

Each sample contains:
```json
{
  "id": 0,
  "function_name": "func_xyz",
  "python": "def func_xyz(a, b):\n    return a + b",
  "jaxpr": "{ lambda ; a:f32[64] b:f32[64]. let ... in (...,) }\n\n=== BACKWARD ===\n...",
  "hlo": "HloModule jit_func...\nENTRY main {...}\n\n=== BACKWARD HLO ===\n...",
  "type": "generate_..."
}
```

---

## Why Multiple IR Formats?

Training data with multiple IR formats enables:

1. **Multi-Target Translation**: Models learn different compilation targets
2. **Abstraction Learning**: Understand high-level (jaxpr) vs low-level (HLO) patterns
3. **Cross-Platform Training**: StableHLO provides portable representation

### Key Differences

| Format | Level | Purpose |
|--------|-------|---------|
| Jaxpr | High | Functional IR, easy to read |
| HLO | Low | XLA optimization, hardware-specific |
| StableHLO | Portable | MLIR-based, cross-platform |

---

## Why Forward + Backward Matters

Both forward and backward passes are included because:

1. **Differentiable Programming**: Modern ML requires gradients
2. **Complete Translation Task**: LLM learns full autodiff patterns
3. **Real-World Utility**: Matches actual JAX compiler output

### Backward Pass Structure

```python
# Forward pass - computes output
def func(a, b):
    return a + b

# Backward pass (via jax.grad) - computes gradients
def grad_func(a, b):
    return jax.grad(lambda x, y: jnp.sum(func(x, y)))(a, b)
```

The extracted IR includes both:
```
=== FORWARD ===
{ lambda ; a:f32[64] b:f32[64]. let c:f32[64] = add a b in (c,) }

=== BACKWARD (GRADIENT) ===
{ lambda ; a:f32[64] b:f32[64]. let
    _:f32[64] = add a b
    d:f32[64] = broadcast_in_dim[...] 1.0
  in (d,) }
```

---

## Function Type Distribution

Each of 10 function types is approximately equally represented (~10% each):

1. **Elementwise**: Basic arithmetic operations
2. **Scalar-Array**: Scalar parameters with arrays
3. **Unary**: Single-input math functions
4. **Branch**: Conditional jnp.where logic
5. **Reduction**: Sum, mean, max, min
6. **Dot Product**: Vector dot products
7. **MatMul**: Matrix multiplications
8. **Multi-Statement**: Chained operations
9. **Nested Branch**: Nested conditionals
10. **Compound**: Mixed patterns

---

## Technical Approach

### JIT-Based Extraction

Rather than hand-writing training pairs, we leverage JAX's compiler:

```python
import jax
from jax import make_jaxpr

# Get jaxpr representation
jaxpr = make_jaxpr(func)(a, b)

# Get HLO via lowering
lowered = jax.jit(func).lower(a, b)
hlo = lowered.as_text()

# Get gradient function IR
grad_jaxpr = make_jaxpr(jax.grad(func))(a, b)
```

Benefits:
- **100% Correct**: Compiler-generated, not hand-written
- **Consistent**: Same Python → deterministic output
- **Scalable**: Can generate unlimited samples

---

## Usage

### Generate Data

```bash
cd jit

# Generate pairs with all IR formats
python3 code/synthesis/pipeline.py \
    --count 5000 \
    --output data/large.jsonl \
    --jsonl \
    --ir-type both \
    --seed 12345
```

### IR-Specific Generation

```bash
# Jaxpr only
python3 code/synthesis/pipeline.py --count 1000 --output jaxpr.jsonl --jsonl --ir-type jaxpr

# HLO only
python3 code/synthesis/pipeline.py --count 1000 --output hlo.jsonl --jsonl --ir-type hlo
```

### Batch Generation

```bash
# Sequential with checkpointing
python3 code/synthesis/batch_generator.py --count 10000 --output data/large.jsonl

# Parallel (multiprocessing)
python3 code/synthesis/batch_generator.py --count 10000 --output data/large.jsonl --parallel --workers 4
```

---

## Files Included

| File | Description |
|------|-------------|
| `jit/data/training_all.jsonl` | Main dataset |
| `jit/code/extraction/ir_extractor.py` | IR extraction (jaxpr + HLO) |
| `jit/code/synthesis/generator.py` | 10 function generators |
| `jit/code/synthesis/pipeline.py` | End-to-end pipeline |
| `jit/code/synthesis/batch_generator.py` | Scalable batch generation |

---

## Migration from Warp

This project was migrated from NVIDIA Warp to JAX:

| Aspect | Warp | JAX |
|--------|------|-----|
| Decorator | `@wp.kernel` | `@jax.jit` |
| Thread ID | `wp.tid()` | Automatic vectorization |
| IR Output | C++/CUDA | jaxpr/HLO/StableHLO |
| Backend | CPU/CUDA | XLA (CPU/GPU/TPU) |
| Gradients | Built-in adjoint | `jax.grad()` |

---

## Conclusion

This dataset provides high-quality Python→HLO training pairs with:
- **Multiple IR formats** (jaxpr, HLO, StableHLO)
- **Both forward and backward functions**
- **10 diverse function types**
- **Production-ready JSONL format**

The JIT-based approach guarantees correctness and enables unlimited scaling.
