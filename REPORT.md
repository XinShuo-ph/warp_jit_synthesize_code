# Technical Report: JAX JIT Code Synthesis Dataset

**Date:** December 2025  
**Purpose:** Python→XLA HLO/MHLO training data for LLM code generation

---

## Executive Summary

Successfully created a pipeline to generate **Python→XLA HLO/MHLO training pairs** using JAX's JIT compilation infrastructure. Each sample contains Python function source paired with compiler-generated intermediate representations including **HLO (High Level Optimizer)**, **optimized HLO**, and **MHLO (MLIR HLO)**, with **forward and backward (gradient) functions**.

---

## Dataset Specifications

| Metric | Value |
|--------|-------|
| Format | JSONL |
| Kernel Types | 10 categories |
| HLO Code | ✅ Forward + Backward |
| Optimized HLO | ✅ With compiler passes |
| MHLO | ✅ MLIR representation |

### Sample Structure

Each sample contains:
```json
{
  "id": 0,
  "kernel_name": "kernel_xyz",
  "python": "def kernel_xyz(...):\n    ...",
  "hlo": "HloModule jit_kernel_xyz...",
  "optimized_hlo": "Optimized HLO IR...",
  "mhlo": "module attributes {...}",
  "type": "generate_..."
}
```

---

## Why XLA HLO?

Training data with HLO enables:

1. **Universal Target**: HLO is used by JAX, TensorFlow, PyTorch, etc.
2. **Hardware-Agnostic**: XLA compiles to CPU, GPU, TPU
3. **Industry Standard**: Widely adopted compiler IR in ML/AI
4. **Optimization Learning**: Models learn compiler optimization patterns

### Key Benefits

| Aspect | Benefit |
|--------|---------|
| Portability | Single IR for multiple backends |
| Performance | Production-grade optimizations |
| Expressiveness | Rich set of operations |
| Ecosystem | Large tooling and community |

---

## Why Forward + Backward Matters

Both forward and backward passes are included because:

1. **Differentiable Programming**: Modern ML requires gradients
2. **Complete Translation Task**: LLM learns full autodiff patterns
3. **Real-World Utility**: Matches actual JAX usage

### Backward Pass Structure

JAX automatically generates backward passes through automatic differentiation:

```python
# Forward pass - computes output
def forward(x):
    return x * 2.0

# Backward pass - computes gradients (automatic via jax.grad)
backward = jax.grad(forward)

# Combined forward + backward in HLO
def combined(x):
    return forward(x), backward(x)
```

The generated HLO contains both the primal computation and the gradient computation in a single module.

---

## Kernel Type Distribution

Each of 10 kernel types is approximately equally represented (~10% each):

1. **Elementwise**: Basic arithmetic operations
2. **Scalar-Array**: Scalar parameters with arrays
3. **Unary**: Single-input math functions
4. **Branch**: Conditional operations with `jnp.where`
5. **Loop**: Iterations using `jax.lax.fori_loop`
6. **Reduction**: Sum, mean, max, min operations
7. **Vector**: Dot products, norms, etc.
8. **Multi-Statement**: Chained operations
9. **Nested Branch**: Nested conditionals
10. **Compound**: Mixed patterns

---

## Technical Approach

### JIT-Based Extraction

Rather than hand-writing training pairs, we leverage JAX's compiler:

```python
import jax

# JIT compile the function
jitted_func = jax.jit(func)

# Lower to HLO
lowered = jitted_func.lower(*example_inputs)

# Get HLO text representation
hlo_text = lowered.as_text()

# Get optimized HLO
compiled = lowered.compile()
optimized_hlo = compiled.as_text()

# Get MHLO (MLIR) representation
mhlo_module = lowered.compiler_ir(dialect='mhlo')
mhlo_text = str(mhlo_module)
```

Benefits:
- **100% Correct**: Compiler-generated, not hand-written
- **Consistent**: Same Python → deterministic output
- **Scalable**: Can generate unlimited samples

---

## Migration from Warp to JAX

This project was migrated from NVIDIA Warp to Google JAX:

### Key Changes

| Component | Warp | JAX |
|-----------|------|-----|
| Decorator | `@wp.kernel` | `@jax.jit` (implicit) |
| Arrays | `wp.array(dtype=float)` | `jnp.array` |
| Thread ID | `wp.tid()` | Vectorized operations |
| Operations | `wp.add`, `wp.mul` | `+`, `*` (native Python) |
| Conditionals | `if/else` | `jnp.where` |
| Loops | `for i in range(n)` | `jax.lax.fori_loop` |
| Atomics | `wp.atomic_add` | Reductions (`jnp.sum`) |
| IR Output | C++/CUDA source | XLA HLO/MHLO |
| Gradients | Built-in adjoint | `jax.grad` |

### Why JAX Over Warp?

1. **Broader Adoption**: JAX is more widely used in ML/AI
2. **Standard IR**: HLO is an industry-standard compiler IR
3. **Flexibility**: Functional programming vs kernel programming
4. **Portability**: Works on CPU, GPU, TPU seamlessly
5. **Ecosystem**: Larger community and library support

---

## Usage

### Generate More Data

```bash
cd jit

# Generate 5000 pairs with all IR representations
python3 code/synthesis/pipeline.py \
    --count 5000 \
    --output data/large.jsonl \
    --jsonl \
    --include-mhlo \
    --seed 12345
```

### Batch Generation

```bash
# Sequential generation with checkpointing
python3 code/synthesis/batch_generator.py \
    --count 10000 \
    --output data/batch.jsonl \
    --seed 42

# Parallel generation (faster)
python3 code/synthesis/batch_generator.py \
    --count 10000 \
    --output data/batch.jsonl \
    --parallel \
    --workers 4
```

---

## Files Included

| File | Description |
|------|-------------|
| `jit/code/extraction/ir_extractor.py` | IR extraction (HLO + MHLO + gradients) |
| `jit/code/synthesis/generator.py` | 10 kernel generators |
| `jit/code/synthesis/pipeline.py` | End-to-end pipeline |
| `jit/code/synthesis/batch_generator.py` | Scalable batch generation |

---

## IR Representations

### HLO (High Level Optimizer)

- Text-based IR used by XLA compiler
- Contains all operations and control flow
- Includes both forward and backward passes

### Optimized HLO

- HLO after compiler optimization passes
- Shows transformations like fusion, CSE, etc.
- Represents actual compiled code structure

### MHLO (MLIR HLO)

- MLIR (Multi-Level Intermediate Representation) dialect
- More structured than text HLO
- Used for advanced compiler transformations

---

## Conclusion

This dataset provides high-quality Python→XLA HLO/MHLO training pairs with:
- **Multiple IR representations** (HLO, optimized HLO, MHLO)
- **Both forward and backward functions**
- **10 diverse kernel types**
- **Production-ready JSONL format**
- **Industry-standard compiler IR**

The JAX-based approach guarantees correctness, enables unlimited scaling, and produces IR that's used across the ML/AI ecosystem.
