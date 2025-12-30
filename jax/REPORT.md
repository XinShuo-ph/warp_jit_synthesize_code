# Technical Report: JAX JIT Code Synthesis Dataset

**Date:** December 2025  
**Purpose:** Python→HLO training data for LLM code generation

---

## Executive Summary

This project generates **Python→HLO/XLA training pairs** using JAX's JIT compilation infrastructure. Each sample contains Python function source paired with compiler-generated code including **forward HLO, backward (gradient) HLO, and optimized HLO**.

---

## Dataset Specifications

| Metric | Value |
|--------|-------|
| Format | JSONL |
| Kernel Types | 10+ categories |
| Forward HLO | ✅ Included |
| Backward HLO | ✅ Gradient computation |
| Optimized HLO | ✅ After XLA passes |

### Sample Structure

Each sample contains:
```json
{
  "id": 0,
  "kernel_name": "kernel_xyz",
  "python": "def kernel_xyz(a, b):\n    return a + b",
  "hlo_forward": "HloModule jit_kernel_xyz...",
  "hlo_backward": "HloModule jit_grad...",
  "hlo_optimized": "HloModule optimized...",
  "type": "generate_..."
}
```

---

## Why JAX and HLO?

### JAX Advantages

1. **XLA Backend**: Industry-standard compiler for ML workloads
2. **Functional Design**: Pure functions enable reliable tracing
3. **Automatic Differentiation**: First-class gradient support
4. **Multi-Platform**: CPU, GPU, TPU from same code

### HLO Benefits for Training

1. **Hardware-Agnostic**: Same IR for all targets
2. **Optimization Visibility**: See fusion, layout decisions
3. **Clean Representation**: Well-structured, parseable format
4. **Production Relevance**: Actual compiler IR used in TensorFlow/JAX

---

## Why Forward + Backward Matters

Both forward and backward passes are included because:

1. **Differentiable Programming**: Modern ML requires gradients
2. **Complete Translation Task**: LLM learns full autodiff patterns
3. **Real-World Utility**: Matches actual JAX compiler behavior

### Forward Pass Structure

```
HloModule jit_fn

ENTRY main {
  x = f32[10] parameter(0)
  two = f32[] constant(2)
  b_two = f32[10] broadcast(two), dimensions={}
  ROOT result = f32[10] multiply(x, b_two)
}
```

### Backward Pass Structure

```
HloModule jit_grad_fn

ENTRY main {
  x = f32[10] parameter(0)
  # Gradient flows backward
  constant = f32[] constant(2)
  broadcast = f32[10] broadcast(constant), dimensions={}
  ROOT grad = f32[10] multiply(upstream_grad, broadcast)
}
```

---

## Kernel Type Distribution

Each of 10+ kernel types is represented:

### Basic Operations
1. **Elementwise**: Basic arithmetic operations
2. **Scalar-Array**: Scalar parameters with arrays
3. **Unary**: Single-input math functions

### Control Flow
4. **Branch**: Conditional jnp.where logic
5. **Loop**: lax.fori_loop iterations
6. **Nested Branch**: Nested conditionals

### Reductions & Vectors
7. **Reduction**: Sum, mean, max operations
8. **Vector**: Dot product, norms

### Complex Patterns
9. **Multi-Statement**: Chained operations
10. **Compound**: Mixed patterns

### ML-Specific (Extended)
11. **MatMul**: Matrix multiplication
12. **Softmax**: Softmax activation
13. **Attention**: Scaled dot-product attention
14. **LayerNorm**: Layer normalization
15. **GELU**: GELU activation
16. **BatchNorm**: Batch normalization

---

## Technical Approach

### JIT-Based Extraction

We leverage JAX's compilation pipeline:

```python
import jax

# Lower to HLO
lowered = jax.jit(fn).lower(*inputs)
hlo_text = lowered.as_text()

# Get optimized HLO
compiled = lowered.compile()
optimized_hlo = compiled.as_text()

# Get gradient HLO
grad_fn = jax.grad(lambda *args: jnp.sum(fn(*args)))
grad_lowered = jax.jit(grad_fn).lower(*inputs)
backward_hlo = grad_lowered.as_text()
```

### Benefits

- **100% Correct**: Compiler-generated, not hand-written
- **Consistent**: Same Python → deterministic output
- **Scalable**: Can generate unlimited samples
- **Authentic**: Uses actual JAX/XLA infrastructure

---

## Comparison: JAX vs Warp

| Feature | JAX (HLO) | Warp (C++/CUDA) |
|---------|-----------|-----------------|
| IR Level | High-level | Low-level |
| Target | XLA backends | CPU/CUDA |
| Fusion | Automatic | Manual |
| Gradient | `jax.grad` | Adjoint |
| Portability | Multi-device | Device-specific |

### When to Use Each

- **JAX/HLO**: Learning compiler IR patterns, XLA optimization
- **Warp/C++**: Learning direct GPU kernel implementation

---

## Usage

### Generate Training Data

```bash
cd jax

# Generate 5000 pairs with forward + backward
python3 code/synthesis/pipeline.py \
    --count 5000 \
    --output data/large.jsonl \
    --jsonl \
    --mode both \
    --seed 12345
```

### Mode-Specific Generation

```bash
# Forward HLO only
python3 code/synthesis/pipeline.py --count 1000 --output forward.jsonl --jsonl --mode forward

# Backward HLO only
python3 code/synthesis/pipeline.py --count 1000 --output backward.jsonl --jsonl --mode backward

# Include ML-specific kernels
python3 code/synthesis/pipeline.py --count 1000 --output ml.jsonl --jsonl --extended
```

---

## Files Included

| File | Description |
|------|-------------|
| `jax/code/extraction/ir_extractor.py` | HLO extraction |
| `jax/code/synthesis/generator.py` | 10+ kernel generators |
| `jax/code/synthesis/pipeline.py` | End-to-end pipeline |
| `jax/notes/jax_basics.md` | JAX compilation flow |
| `jax/notes/hlo_format.md` | HLO structure docs |

---

## Conclusion

This dataset provides high-quality Python→HLO training pairs with:
- **Forward, backward, and optimized HLO**
- **10+ diverse kernel types**
- **ML-specific extended generators**
- **Production-ready JSONL format**

The JIT-based approach guarantees correctness and enables unlimited scaling using JAX's robust compilation infrastructure.
