# Technical Report: JAX JIT Code Synthesis Dataset

**Date:** December 2025  
**Purpose:** Python→HLO/XLA training data for LLM code generation

---

## Executive Summary

Successfully migrated the training data generation pipeline from NVIDIA Warp to **JAX** (Google's high-performance numerical computing library). The pipeline now generates **Python→HLO** training pairs using JAX's XLA compiler infrastructure. Each sample contains Python function source paired with compiler-generated **JAXPR** (JAX's intermediate representation) and **HLO** (High-Level Optimizer IR) code, including **forward and backward (gradient) functions**.

---

## Dataset Specifications

| Metric | Value |
|--------|-------|
| Framework | JAX (XLA compiler) |
| Format | JSONL |
| Function Types | 10 categories |
| JAXPR Code | ✅ Intermediate representation |
| HLO Code | ✅ Optimized low-level IR |
| Autodiff | ✅ Automatic differentiation |

### Sample Structure

Each sample contains:
```json
{
  "id": 0,
  "function_name": "function_xyz",
  "python": "def function_xyz(...):\n    ...",
  "jaxpr": "{ lambda ; a:f32[4]. let ... }",
  "hlo": "HloModule jit_function_xyz\n...",
  "type": "generate_..."
}
```

---

## Why JAX?

JAX offers several advantages over Warp for training data generation:

1. **Automatic Differentiation**: Built-in grad transformation for automatic gradient computation
2. **Multiple Backends**: XLA compiles for CPU, GPU, and TPU automatically
3. **NumPy Compatibility**: Familiar API reduces learning curve
4. **Mature Ecosystem**: Well-documented, actively developed by Google
5. **Interpretable IR**: JAXPR provides human-readable intermediate representation

### Key JAX Features

| Feature | Description |
|---------|-------------|
| `jax.jit` | JIT compilation using XLA |
| `jax.grad` | Automatic differentiation |
| `jax.lax` | Low-level operations (scan, cond, etc.) |
| JAXPR | Intermediate representation (interpretable) |
| HLO | High-Level Optimizer IR (optimized) |

---

## Why Forward + Backward Matters

Both forward and backward passes are included because:

1. **Differentiable Programming**: Modern ML requires gradients
2. **Complete Translation Task**: LLM learns full autodiff patterns
3. **Automatic Derivation**: JAX automatically generates gradient code

### Gradient Computation Example

```python
# Forward function
def forward(x):
    return x * 2.0 + 1.0

# Backward (gradient) - automatically derived
grad_fn = jax.grad(forward)

# JAXPR shows both forward and backward computation
jaxpr = make_jaxpr(grad_fn)(1.0)
```

JAX's autodiff generates efficient gradient code through:
- Reverse-mode automatic differentiation
- Efficient backpropagation through computation graph
- Optimized memory usage with XLA

---

## Function Type Distribution

Each of 10 function types is approximately equally represented (~10% each):

1. **Elementwise**: Basic arithmetic operations
2. **Scalar-Array**: Scalar parameters with arrays
3. **Unary**: Single-input math functions
4. **Branch**: Conditional operations (jnp.where)
5. **Loop**: Scan-based iterations
6. **Reduction**: Sum, mean, etc.
7. **Vector**: Vector norm, dot product, etc.
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

# Get JAXPR (intermediate representation)
jaxpr = make_jaxpr(func)(*args)

# Get HLO (optimized representation)
lowered = jax.jit(func).lower(*args)
hlo_code = lowered.as_text()

# Get gradient function
grad_fn = jax.grad(func)
grad_jaxpr = make_jaxpr(grad_fn)(*args)
```

Benefits:
- **100% Correct**: Compiler-generated, not hand-written
- **Consistent**: Same Python → deterministic output
- **Scalable**: Can generate unlimited samples
- **Device Agnostic**: XLA handles CPU/GPU/TPU compilation

---

## Migration from Warp to JAX

### Key Changes

| Aspect | Warp | JAX |
|--------|------|-----|
| Decorator | `@wp.kernel` | `@jax.jit` |
| Arrays | `wp.array(dtype=float)` | `jnp.ndarray` |
| Threading | `tid = wp.tid()` | Vectorized operations |
| Conditionals | `if/else` | `jnp.where()` |
| Loops | `for i in range(n)` | `jax.lax.scan()` |
| Atomics | `wp.atomic_add()` | `jnp.sum()` |
| Output | C++/CUDA code | JAXPR/HLO code |

### Function Examples

**Warp (before)**:
```python
@wp.kernel
def saxpy(a: float, x: wp.array(dtype=float), y: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a * x[tid] + y[tid]
```

**JAX (after)**:
```python
def saxpy(a, x, y):
    """SAXPY operation."""
    return a * x + y
```

---

## Usage

### Generate More Data

```bash
cd jit

# Generate 5000 pairs
python3 code/synthesis/pipeline.py \
    --count 5000 \
    --output data/large.jsonl \
    --jsonl \
    --seed 12345
```

### Device-Specific Generation

JAX automatically compiles for available hardware, but you can specify preferences:

```bash
# Default (uses available hardware)
python3 code/synthesis/pipeline.py --count 1000 --output data.jsonl --jsonl

# CPU-focused
JAX_PLATFORMS=cpu python3 code/synthesis/pipeline.py --count 1000 --output cpu.jsonl --jsonl --device cpu
```

---

## Files Included

| File | Description |
|------|-------------|
| `jit/data/training_all.jsonl` | Main dataset |
| `jit/code/extraction/ir_extractor.py` | IR extraction (JAXPR + HLO) |
| `jit/code/synthesis/generator.py` | 10 function generators |
| `jit/code/synthesis/pipeline.py` | End-to-end pipeline |

---

## Advantages of JAX-based Approach

1. **Simplified Code**: No manual threading/indexing required
2. **Better Autodiff**: Automatic gradient computation
3. **Multiple Backends**: CPU, GPU, TPU support
4. **NumPy Compatibility**: Easier to understand and maintain
5. **Production Ready**: Used in production at Google, DeepMind, etc.
6. **Interpretable IR**: JAXPR is more readable than C++/CUDA

---

## Conclusion

This dataset provides high-quality Python→HLO training pairs with:
- **JAXPR and HLO representations**
- **Automatic gradient computation**
- **10 diverse function types**
- **Production-ready JSONL format**

The JAX-based approach guarantees correctness, enables unlimited scaling, and provides both interpretable (JAXPR) and optimized (HLO) representations for comprehensive LLM training.
