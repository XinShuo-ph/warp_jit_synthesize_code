# Technical Report: JIT Code Synthesis Dataset

**Date:** December 2025  
**Purpose:** Python→C++/CUDA training data for LLM code generation

---

## Executive Summary

Successfully generated **1,500 Python→C++/CUDA training pairs** (18MB) using NVIDIA Warp's JIT compilation infrastructure. Each sample contains Python kernel source paired with compiler-generated code for **both CPU and CUDA backends**, each including **forward and backward (gradient) functions**.

---

## Dataset Specifications

| Metric | Value |
|--------|-------|
| Total Samples | 1,500 |
| File Size | 18 MB |
| Format | JSONL |
| Kernel Types | 10 categories |
| CPU Code | ✅ Forward + Backward |
| CUDA Code | ✅ Forward + Backward |

### Sample Structure

Each sample contains:
```json
{
  "id": 0,
  "kernel_name": "kernel_xyz",
  "python": "@wp.kernel\ndef kernel_xyz(...):\n    ...",
  "cpp": "void kernel_cpu_kernel_forward(...) {...}\nvoid kernel_cpu_kernel_backward(...) {...}",
  "cuda": "void kernel_cuda_kernel_forward(...) {...}\nvoid kernel_cuda_kernel_backward(...) {...}",
  "type": "generate_..."
}
```

---

## Why Both CPU and CUDA?

Training data with both backends enables:

1. **Backend-Aware Translation**: Models learn hardware-specific patterns
2. **Cross-Compilation Training**: Same Python → different targets
3. **Optimization Learning**: Understand CPU vs GPU trade-offs

### Key Differences

| Aspect | CPU | CUDA |
|--------|-----|------|
| Execution | Sequential loop | Parallel threads |
| Thread ID | `task_index` | `blockIdx * blockDim + threadIdx` |
| Args | Struct pointer | Direct parameters |
| Memory | Stack-based | Shared memory support |

---

## Why Forward + Backward Matters

Both forward and backward passes are included because:

1. **Differentiable Programming**: Modern ML requires gradients
2. **Complete Translation Task**: LLM learns full autodiff patterns
3. **Real-World Utility**: Matches actual Warp compiler output

### Backward Pass Structure

```cpp
// Forward pass - computes output
void kernel_cuda_kernel_forward(...) {
    var_0 = builtin_tid1d();
    var_1 = wp::load(var_a, var_0);
    var_2 = wp::mul(var_1, 2.0);
    wp::array_store(var_b, var_0, var_2);
}

// Backward pass - computes gradients  
void kernel_cuda_kernel_backward(...) {
    // Replay forward
    var_0 = builtin_tid1d();
    var_1 = wp::load(var_a, var_0);
    var_2 = wp::mul(var_1, 2.0);
    
    // Reverse-mode autodiff
    wp::adj_array_store(...);
    wp::adj_mul(var_1, 2.0, adj_1, adj_const, adj_2);
    wp::adj_load(var_a, var_0, adj_a, adj_0, adj_1);
}
```

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

Rather than hand-writing training pairs, we leverage Warp's compiler:

```python
import warp._src.context as ctx

# Create code builder
hasher = ctx.ModuleHasher(module)
builder = ctx.ModuleBuilder(module, options, hasher)

# Generate CPU code (forward + backward)
cpp_code = builder.codegen("cpu")

# Generate CUDA code (forward + backward)
cuda_code = builder.codegen("cuda")
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

**Solution**: Used improved `ir_extractor.py` from branch 6a68's approach to generate both forward and backward for both CPU and CUDA.

---

## Usage

### Generate More Data

```bash
cd jit

# Generate 5000 pairs with both CPU and CUDA
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

This dataset provides high-quality Python→C++/CUDA training pairs with:
- **Both CPU and CUDA backends**
- **Both forward and backward functions**
- **10 diverse kernel types**
- **Production-ready JSONL format**

The JIT-based approach guarantees correctness and enables unlimited scaling.
