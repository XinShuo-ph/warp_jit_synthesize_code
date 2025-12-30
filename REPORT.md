# Technical Report: JAX JIT Code Synthesis Dataset

**Date:** December 2025  
**Purpose:** Python→XLA HLO training data for LLM code generation

---

## Executive Summary

Successfully migrated code synthesis pipeline to **JAX** for generating Python→XLA HLO training pairs. Each sample contains Python function source paired with compiler-generated **XLA HLO (High-Level Operations)** representation, including **forward and backward (gradient) functions**.

---

## Dataset Specifications

| Metric | Value |
|--------|-------|
| Format | JSONL |
| Function Types | 10 categories |
| HLO Code | ✅ Forward + Backward |
| Optimized HLO | ✅ When available |

### Sample Structure

Each sample contains:
```json
{
  "id": 0,
  "kernel_name": "kernel_xyz",
  "python": "def kernel_xyz(a, b):\n    \"\"\"Elementwise operation.\"\"\"\n    return a + b",
  "hlo": "HloModule with forward and backward passes",
  "optimized_hlo": "Optimized XLA HLO code",
  "type": "generate_..."
}
```

---

## Why XLA HLO?

Training data with XLA HLO enables:

1. **Device-Agnostic Code**: HLO compiles to CPU, GPU, TPU
2. **Industry Standard**: Used by JAX, TensorFlow, PyTorch 2.0+
3. **Optimization Learning**: Understand compiler optimizations
4. **Modern ML Focus**: Aligns with current deep learning frameworks

### Key Features

| Aspect | XLA HLO |
|--------|---------|
| Representation | Device-agnostic IR |
| Execution | Compiled to target hardware |
| Memory | XLA-managed, optimized layout |
| Gradients | Automatic differentiation |

---

## Why Forward + Backward Matters

Both forward and backward passes are included because:

1. **Differentiable Programming**: Modern ML requires gradients
2. **Complete Translation Task**: LLM learns full autodiff patterns
3. **Real-World Utility**: Matches actual JAX/XLA compiler output

### Backward Pass with JAX

```python
# Forward pass function
def forward(a, b):
    return a * 2.0 + b

# JAX automatically generates gradient function
grad_fn = jax.grad(forward)

# Compiled to XLA HLO with both forward and backward
```

XLA HLO includes:
- Forward computation graph
- Automatic differentiation (reverse-mode)
- Optimized gradient computation

---

## Function Type Distribution

Each of 10 function types is approximately equally represented:

1. **Elementwise**: Basic arithmetic operations
2. **Scalar-Array**: Scalar parameters with arrays
3. **Unary**: Single-input math functions
4. **Branch**: Conditional logic with `jnp.where`
5. **Loop**: Scan/reduction operations
6. **Reduction**: Sum, mean, etc.
7. **Vector**: Dot products, norms
8. **Multi-Statement**: Chained operations
9. **Nested Branch**: Nested conditionals
10. **Compound**: Mixed patterns

---

## Technical Approach

### JIT-Based Extraction with JAX

Rather than hand-writing training pairs, we leverage JAX's compiler:

```python
import jax
import jax.numpy as jnp
from jax import grad, jit

# Define function
def my_function(a, b):
    return a * 2.0 + b

# Lower to XLA HLO
lowered = jax.jit(my_function).lower(sample_a, sample_b)
hlo_text = lowered.as_text()

# Get optimized HLO
compiled = lowered.compile()
optimized_hlo = compiled.as_text()

# Generate gradient function
grad_fn = grad(lambda a, b: jnp.sum(my_function(a, b)))
grad_lowered = jax.jit(grad_fn).lower(sample_a, sample_b)
grad_hlo = grad_lowered.as_text()
```

Benefits:
- **100% Correct**: Compiler-generated, not hand-written
- **Consistent**: Same Python → deterministic HLO output
- **Scalable**: Can generate unlimited samples
- **Standard IR**: XLA HLO used across JAX, TensorFlow, PyTorch

---

## Migration from Warp to JAX

### Why JAX?

1. **Industry Standard**: XLA HLO used by JAX, TensorFlow, PyTorch 2.0+
2. **Device Agnostic**: Single IR compiles to CPU, GPU, TPU
3. **Modern Ecosystem**: Better integration with current ML frameworks
4. **Composability**: Transformations like `jit`, `grad`, `vmap` are built-in
5. **Wider Adoption**: More relevant for contemporary ML development

### Key Changes

| Aspect | Warp | JAX |
|--------|------|-----|
| IR Format | C++/CUDA code | XLA HLO |
| Backend | CPU and CUDA separately | Device-agnostic |
| Syntax | `@wp.kernel` decorator | Standard Python functions |
| Arrays | `wp.array()` | `jnp.ndarray` |
| Gradients | `kernel.adj` | `jax.grad()` |
| Compilation | `wp.launch()` | `jax.jit()` |

---

## Usage

### Generate Training Data

```bash
cd jit

# Generate 1000 pairs with XLA HLO
python3 code/synthesis/pipeline.py \
    --count 1000 \
    --output data/training.jsonl \
    --jsonl \
    --seed 12345
```

### Test Examples

```bash
# Test simple addition
python3 jit/code/examples/test_add_kernel.py

# Test SAXPY (scalar * x + y)
python3 jit/code/examples/test_saxpy.py

# Test dot product
python3 jit/code/examples/test_dot_product.py

# Test IR extraction
python3 jit/code/extraction/ir_extractor.py
python3 jit/code/extraction/test_ir_extractor.py
```

---

## Files Included

| File | Description |
|------|-------------|
| `jit/code/extraction/ir_extractor.py` | XLA HLO extraction from JAX |
| `jit/code/extraction/test_ir_extractor.py` | Test IR extraction |
| `jit/code/synthesis/generator.py` | 10 function generators |
| `jit/code/synthesis/pipeline.py` | End-to-end pipeline |
| `jit/code/examples/test_*.py` | Example JAX functions |

---

## Conclusion

This migrated dataset provides high-quality Python→XLA HLO training pairs with:
- **XLA HLO intermediate representation** (industry standard)
- **Forward and backward functions** (automatic differentiation)
- **10 diverse function types**
- **Device-agnostic compilation** (CPU, GPU, TPU)
- **Production-ready JSONL format**

The JAX-based approach guarantees correctness, enables unlimited scaling, and produces IR that's relevant across modern ML frameworks.
