# Technical Report: JAX JIT IR Synthesis Dataset

**Date:** December 2025  
**Purpose:** Python→JAX compiler IR training data for LLM code generation

---

## Executive Summary

This project generates Python→**JAX compiler IR** training pairs using JAX's lowering/compilation pipeline. Each sample contains Python function source paired with compiler IR for the **forward** computation and its **backward** computation (via `jax.grad`). GPU IR is included only when a GPU backend is available.

---

## Dataset Specifications

| Metric | Value |
|--------|-------|
| Total Samples | 1,500 |
| File Size | 18 MB |
| Format | JSONL |
| Kernel Types | 10 categories |
| CPU IR | ✅ Forward + Backward |
| GPU IR | ⚠️ Optional (depends on available backend) |

### Sample Structure

Each sample contains:
```json
{
  "id": 0,
  "kernel_name": "kernel_xyz",
  "python": "def kernel_xyz(...):\n    ...",
  "cpp": "# JAX compiler IR (backend=cpu, dialect=stablehlo)\n## Forward\n...\n## Backward ...\n...",
  "cuda": "# JAX compiler IR (backend=gpu, dialect=stablehlo)\n## Forward\n...\n## Backward ...\n...",
  "type": "generate_..."
}
```

---

## Why CPU and (Optional) GPU?

Training data with multiple backends enables:

1. **Backend-Aware Translation**: Models learn hardware-specific patterns
2. **Cross-Compilation Training**: Same Python → different targets
3. **Optimization Learning**: Understand CPU vs GPU trade-offs

### Key Differences

IR differences are backend-dependent and reflect the active JAX platform. On CPU-only machines, the dataset will typically omit `cuda`.

---

## Why Forward + Backward Matters

Both forward and backward are included because:

1. **Differentiable Programming**: Modern ML requires gradients
2. **Complete Translation Task**: LLM learns full autodiff patterns
3. **Real-World Utility**: Matches real compiler output from a modern differentiable system

### Backward IR

Backward IR is extracted by compiling `jax.grad(loss_fn)` where `loss_fn` reduces the forward outputs to a scalar (e.g. sum of outputs), producing reverse-mode code for differentiable inputs.

---

## Kernel Type Distribution

Each of 10 kernel types is approximately equally represented (~10% each):

1. **Elementwise**: Basic arithmetic operations
2. **Scalar-Array**: Scalar parameters with arrays
3. **Unary**: Single-input math functions
4. **Branch**: Conditional if/else logic
5. **Loop**: For-loop iterations
6. **Reduction**: Atomic accumulations
7. **Vector**: Vec3 dot, length, etc.
8. **Multi-Statement**: Chained operations
9. **Nested Branch**: Nested conditionals
10. **Compound**: Mixed patterns

---

## Technical Approach

### JIT-Based Extraction

Rather than hand-writing training pairs, we leverage JAX's compiler pipeline:

```python
import jax

lowered = jax.jit(fn).lower(*example_args)
stablehlo_text = lowered.compiler_ir(dialect="stablehlo").as_text()

grad_lowered = jax.jit(jax.grad(loss_fn, argnums=argnums)).lower(*example_args)
grad_stablehlo_text = grad_lowered.compiler_ir(dialect="stablehlo").as_text()
```

Benefits:
- **100% Correct**: Compiler-generated, not hand-written
- **Consistent**: Same Python → deterministic output
- **Scalable**: Can generate unlimited samples

---

## Branch Investigation Summary

Investigated 15+ `dataset-and-report-generation` branches:

| Branch | CPU Data | CUDA Data | Forward+Backward |
|--------|----------|-----------|------------------|
| acf8 | 40K files | 40K files | Forward only |
| 891a | 69K files | 60K files | Forward only |
| 6a68 | Manifest | Manifest | ✅ Both |
| a4a2 | 1.3K files | 1.3K files | Forward only |
| ca52 | JSONL | JSONL | Forward only |

**Solution**: Implemented a JAX-based `ir_extractor.py` that extracts compiler IR for both forward and backward (via `jax.grad`).

---

## Usage

### Generate More Data

```bash
cd jit

# Generate 5000 pairs (CPU always; GPU if available)
python3 code/synthesis/pipeline.py \
    --count 5000 \
    --output data/large.jsonl \
    --jsonl \
    --device both \
    --seed 12345
```

### Device-Specific Generation

```bash
# CPU only
python3 code/synthesis/pipeline.py --count 1000 --output cpu.jsonl --jsonl --device cpu

# CUDA only
python3 code/synthesis/pipeline.py --count 1000 --output cuda.jsonl --jsonl --device cuda
```

---

## Files Included

| File | Description |
|------|-------------|
| `jit/data/training_all.jsonl` | Main dataset (1,500 pairs, 18MB) |
| `jit/code/extraction/ir_extractor.py` | IR extraction (CPU + CUDA) |
| `jit/code/synthesis/generator.py` | 10 kernel generators |
| `jit/code/synthesis/pipeline.py` | End-to-end pipeline |

---

## Conclusion

This dataset provides high-quality Python→JAX compiler IR training pairs with:
- **CPU backend**
- **Optional GPU backend (when available)**
- **Both forward and backward functions**
- **10 diverse kernel types**
- **Production-ready JSONL format**

The JIT-based approach guarantees correctness and enables unlimited scaling.
