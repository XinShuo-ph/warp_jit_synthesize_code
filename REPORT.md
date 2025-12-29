# Technical Report: JIT Code Synthesis Dataset

**Date:** December 2025  
**Purpose:** Python→C++ training data for LLM code generation

---

## Executive Summary

Successfully generated **1,505 Python→C++ training pairs** (8.5MB) using NVIDIA Warp's JIT compilation infrastructure. Each sample contains Python kernel source paired with the compiler-generated C++ code including **both forward and backward (gradient) functions**.

---

## Dataset Specifications

| Metric | Value |
|--------|-------|
| Total Samples | 1,505 |
| File Size | 8.5 MB |
| Format | JSONL |
| Kernel Types | 10 categories |
| Forward Pass | ✅ Included |
| Backward Pass | ✅ Included |

### Kernel Type Distribution

Each of 10 kernel types is approximately equally represented (~10% each):
- Elementwise operations
- Scalar-array operations  
- Unary math functions
- Conditional branching
- For loops
- Atomic reductions
- Vector operations
- Multi-statement kernels
- Nested conditionals
- Compound patterns

---

## Why Forward + Backward Matters

The dataset includes **both forward and backward passes** in the C++ code. This is critical because:

1. **Differentiable Programming**: Modern ML frameworks require gradient computation
2. **Complete Translation Task**: LLM must learn to generate autodiff code
3. **Real-World Utility**: Matches actual compiler output structure

### Example Structure

```cpp
// Forward pass - computes output
void kernel_cpu_kernel_forward(...) {
    var_0 = builtin_tid1d();
    var_1 = wp::load(var_a, var_0);
    var_2 = wp::mul(var_1, 2.0);
    wp::array_store(var_b, var_0, var_2);
}

// Backward pass - computes gradients
void kernel_cpu_kernel_backward(...) {
    // Replay forward to get intermediate values
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

## Technical Approach

### JIT-Based Data Generation

Rather than manually writing training pairs, we leverage the Warp compiler:

1. **Generate** random Python kernels programmatically
2. **Compile** using Warp's JIT infrastructure
3. **Extract** the generated C++ via internal APIs
4. **Pair** the original Python with extracted C++

This ensures:
- **100% correctness**: C++ is compiler-generated, not hand-written
- **Consistency**: Same Python always produces same C++
- **Scalability**: Can generate unlimited samples

### IR Extraction Method

```python
from warp._src.codegen import codegen_kernel, codegen_module

# Trigger compilation
kernel.module.load("cpu")

# Extract C++ code
cpp_kernel = codegen_kernel(kernel, "cpu", options)
cpp_module = codegen_module(kernel, "cpu", options)
```

---

## Use Cases

1. **Code Translation Models**: Train Python→C++ translators
2. **Compiler Assistants**: Learn to explain/optimize IR
3. **Autodiff Training**: Teach models to generate gradient code
4. **GPU Code Understanding**: Learn parallel programming patterns

---

## Quality Validation

- **JSON Validity**: 100% parseable
- **Python Syntax**: All kernels compile successfully
- **C++ Completeness**: Contains struct, forward, backward, and entry points
- **Type Safety**: Properly typed Warp types (float32, vec3, etc.)

---

## Scaling Recommendations

To generate more data:

```bash
# Generate 10,000 pairs
python3 jit/code/synthesis/batch_generator.py --count 10000 --output data/large.jsonl

# Different seeds for variety
python3 jit/code/synthesis/pipeline.py --count 5000 --seed 12345 --output data/batch2.jsonl --jsonl
```

Expected rate: ~0.8-1.0 pairs/second (CPU-only mode)

---

## Files Included

| File | Description |
|------|-------------|
| `jit/data/training_all.jsonl` | Main dataset (1,505 pairs) |
| `jit/code/extraction/ir_extractor.py` | IR extraction utility |
| `jit/code/synthesis/generator.py` | 10 kernel generators |
| `jit/code/synthesis/pipeline.py` | End-to-end pipeline |
| `jit/code/synthesis/batch_generator.py` | Scalable generation |

---

## Conclusion

This dataset provides high-quality Python→C++ training pairs with both forward and backward passes included. The JIT-based approach guarantees correctness and enables unlimited scaling. The 10 kernel type categories ensure diverse coverage of common GPU programming patterns.
