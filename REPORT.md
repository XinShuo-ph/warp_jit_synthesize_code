# Technical Report: JAX JIT Code Synthesis Dataset

**Date:** December 2025  
**Purpose:** Python→HLO/XLA training data for LLM code generation

---

## Executive Summary

Successfully migrated the code synthesis pipeline from NVIDIA Warp to **JAX**. The system generates **Python→HLO training pairs** using JAX's JIT compilation infrastructure. Each sample contains Python function source paired with compiler-generated code including **HLO, Jaxpr, and Optimized HLO**, each including **forward and backward (gradient) functions**.

---

## Dataset Specifications

| Metric | Value |
|--------|-------|
| Format | JSONL |
| Function Types | 11 categories |
| HLO Code | ✅ Forward + Backward |
| Jaxpr Code | ✅ Forward + Backward |
| Optimized HLO | ✅ When available |

### Sample Structure

Each sample contains:
```json
{
  "id": 0,
  "kernel_name": "kernel_xyz",
  "python": "def kernel_xyz(a, b):\n    return a + b",
  "hlo": "HloModule jit_kernel_xyz { ... }",
  "jaxpr": "{ lambda ; a:f32[100]. let ... in (...) }",
  "optimized_hlo": "HloModule jit_kernel_xyz, ... optimized ...",
  "type": "generate_..."
}
```

---

## Why JAX?

JAX provides several advantages for this code synthesis task:

1. **XLA Backend**: Direct access to Google's XLA compiler infrastructure
2. **HLO Representation**: Industry-standard intermediate representation
3. **Automatic Differentiation**: Built-in gradient computation
4. **Cross-Platform**: Supports CPU, GPU, and TPU
5. **Wide Adoption**: Used in major ML frameworks (Flax, Haiku, etc.)

---

## IR Types Explained

### Jaxpr (JAX Program Representation)

High-level functional representation of the computation:

```
{ lambda ; a:f32[100] b:f32[100]. let
    c:f32[100] = add a b
    d:f32[100] = mul c 2.0
  in (d,) }
```

Benefits:
- Platform-independent
- Shows JAX primitive operations
- Easy to understand and analyze

### HLO (High Level Optimizer)

XLA's intermediate representation:

```
HloModule jit_func, entry_computation_layout={(f32[100]{0}, f32[100]{0})->f32[100]{0}}

ENTRY main.4 {
  Arg_0.1 = f32[100]{0} parameter(0)
  Arg_1.2 = f32[100]{0} parameter(1)
  add.3 = f32[100]{0} add(Arg_0.1, Arg_1.2)
  ROOT multiply.4 = f32[100]{0} multiply(add.3, constant.5)
}
```

Benefits:
- Lower-level representation
- Shows actual XLA operations
- Platform-specific details

### Optimized HLO

After XLA compilation with all optimizations:

- Operation fusion
- Memory layout optimizations
- Platform-specific code generation

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
    return a * 2.0 + b

# Backward pass - computes gradients
# JAX automatically generates:
{ lambda ; a:f32[100] b:f32[100]. let
    # Forward replay
    c = mul a 2.0
    d = add c b
    # Reverse-mode autodiff
    grad_d = ones_like(d)  # upstream gradient
    grad_c = grad_d
    grad_a = mul grad_c 2.0
    grad_b = grad_d
  in (grad_a,) }
```

---

## Function Type Distribution

Each of 11 function types is approximately equally represented:

1. **Elementwise**: Basic arithmetic operations
2. **Scalar-Array**: Scalar parameters with arrays
3. **Unary**: Single-input math functions (sin, cos, sqrt, etc.)
4. **Branch**: Conditional operations (jnp.where)
5. **Loop**: Iterative computations (jax.lax.fori_loop)
6. **Reduction**: Sum, mean, max, etc.
7. **Vector**: Dot products, norms
8. **Multi-Statement**: Chained operations
9. **Nested Branch**: Nested conditionals
10. **Compound**: Mixed patterns
11. **Matrix Multiplication**: matmul operations

---

## Technical Approach

### JIT-Based Extraction

We leverage JAX's compilation infrastructure:

```python
import jax
import jax.numpy as jnp

def func(a, b):
    return a * 2.0 + b

# Get Jaxpr (JAX's IR)
jaxpr = jax.make_jaxpr(func)(sample_a, sample_b)

# Get HLO (before optimization)
lowered = jax.jit(func).lower(sample_a, sample_b)
hlo_text = lowered.as_text()

# Get Optimized HLO (after XLA compilation)
compiled = lowered.compile()
optimized_hlo = compiled.as_text()

# Get gradient function IR
grad_fn = jax.grad(lambda a, b: jnp.sum(func(a, b)))
grad_jaxpr = jax.make_jaxpr(grad_fn)(sample_a, sample_b)
```

Benefits:
- **100% Correct**: Compiler-generated, not hand-written
- **Consistent**: Same Python → deterministic output
- **Scalable**: Can generate unlimited samples

---

## Migration from Warp

| Aspect | Warp (Previous) | JAX (Current) |
|--------|-----------------|---------------|
| Backend | NVIDIA Warp | Google JAX/XLA |
| IR Format | C++/CUDA | HLO/Jaxpr |
| GPU Support | NVIDIA only | NVIDIA, AMD, TPU |
| Compilation | Warp compiler | XLA compiler |
| Gradients | wp.adj_* functions | jax.grad |

---

## Usage

### Generate Training Data

```bash
cd jit

# Generate pairs with all IR types
python3 code/synthesis/pipeline.py \
    --count 5000 \
    --output data/large.jsonl \
    --jsonl \
    --ir-type both \
    --seed 12345
```

### IR-Type Specific Generation

```bash
# HLO only
python3 code/synthesis/pipeline.py --count 1000 --output hlo.jsonl --jsonl --ir-type hlo

# Jaxpr only
python3 code/synthesis/pipeline.py --count 1000 --output jaxpr.jsonl --jsonl --ir-type jaxpr

# Optimized HLO only
python3 code/synthesis/pipeline.py --count 1000 --output opt.jsonl --jsonl --ir-type optimized
```

---

## Files Included

| File | Description |
|------|-------------|
| `jit/data/training_all.jsonl` | Main dataset |
| `jit/code/extraction/ir_extractor.py` | IR extraction (HLO + Jaxpr) |
| `jit/code/synthesis/generator.py` | 11 function generators |
| `jit/code/synthesis/pipeline.py` | End-to-end pipeline |
| `jit/code/synthesis/batch_generator.py` | Scalable batch generation |

---

## Conclusion

This dataset provides high-quality Python→HLO training pairs with:
- **Multiple IR formats** (HLO, Jaxpr, Optimized HLO)
- **Both forward and backward functions**
- **11 diverse function types**
- **Production-ready JSONL format**

The JAX-based approach guarantees correctness and enables unlimited scaling, with the added benefit of supporting multiple hardware backends (CPU, GPU, TPU).
