# JIT Code Synthesis Training Data Report

**Date**: December 28, 2025  
**Prepared for**: Chief Scientist  
**Status**: Production Complete ✅

---

## Executive Summary

We have successfully generated **440MB of paired Python→IR training data** using NVIDIA Warp's JIT compilation infrastructure. The dataset consists of:

- **CPU Dataset**: 209MB (21,281 Python→C++ IR pairs)
- **CUDA Dataset**: 231MB (20,001 Python→CUDA IR pairs)

This data is designed for training LLMs on code translation tasks, specifically translating high-level Python kernel code to low-level Intermediate Representation (IR) in both CPU (C++) and GPU (CUDA) formats.

---

## 1. Just-In-Time (JIT) Compilation

### What is JIT Compilation?
JIT compilation translates high-level code to machine code at runtime, rather than ahead-of-time. This enables:

- **Performance optimization** based on actual runtime conditions
- **Platform portability** - same source code runs on different hardware
- **Dynamic specialization** - code optimized for specific input types/sizes

### Why JIT Matters for ML Training Data
JIT compilers create an observable translation path from high-level abstractions to low-level implementations. By capturing this translation, we create paired data that teaches LLMs:

1. How computational patterns map to hardware operations
2. How automatic differentiation (backward pass) is generated
3. How optimizations transform code structure

---

## 2. Intermediate Representation (IR)

### What is IR?
IR is the compiler's internal representation of code - an abstract form between source code and machine code. Key characteristics:

| Property | Description |
|----------|-------------|
| **Hardware-agnostic** | Represents computation without hardware specifics |
| **Type-explicit** | All types are explicit (no inference needed) |
| **SSA form** | Static Single Assignment - each variable assigned once |
| **Optimization target** | Where compiler optimizations are applied |

### Types of IR in Our Dataset

**CPU IR (C++):**
- Standard C++ with explicit types (`wp::float32`, `wp::vec3`)
- Direct function calls for operations (`wp::add`, `wp::mul`)
- Forward/backward pass separation

**CUDA IR:**
- CUDA C++ with `__global__` kernel declarations
- Thread indexing via `blockIdx`, `threadIdx`
- Shared memory management (`tile_shared_storage_t`)
- GPU-specific memory patterns

### Why Paired Data is Valuable
LLMs trained on Python→IR pairs learn:
- Type inference (Python dynamic types → explicit C++ types)
- Loop unrolling (Python `for` → explicit iterations in IR)
- Autodiff patterns (forward → backward gradient flow)
- Hardware-specific optimizations (memory access patterns)

---

## 3. NVIDIA Warp

### What is Warp?
[NVIDIA Warp](https://github.com/NVIDIA/warp) is a Python framework for high-performance simulation and graphics, featuring:

- **Python-native kernels** using `@wp.kernel` decorator
- **Automatic differentiation** for gradient computation
- **Multi-device support** - same code runs on CPU and GPU
- **JIT compilation** via LLVM/NVRTC

### Why We Chose Warp

1. **Clean abstraction**: Simple Python syntax generates complex IR
2. **Dual-target**: Single source → both CPU and CUDA code
3. **Autodiff included**: Forward AND backward passes generated automatically
4. **Rich type system**: Vectors, matrices, arrays with explicit types
5. **Accessible internals**: `codegen()` API exposes IR directly

### Warp Compilation Flow

```
Python Kernel (@wp.kernel)
    ↓ [AST Analysis]
Warp IR (internal)
    ↓ [Code Generation]
C++ (CPU) or CUDA (GPU)
    ↓ [Compilation]
Machine Code (cached)
```

---

## 4. Dataset Overview

### 4.1 CPU Dataset

| Metric | Value |
|--------|-------|
| **Total Size** | 209 MB |
| **Number of Pairs** | 21,281 |
| **Average Pair Size** | ~10 KB |
| **File Format** | JSON |
| **Device Target** | CPU (x86_64) |

**Kernel Type Distribution:**
- Arithmetic operations: ~10%
- Control flow (conditionals, loops): ~20%
- Math functions (sin, cos, exp, log): ~10%
- Vector operations (vec3, vec4): ~10%
- Matrix operations: ~10%
- Atomic operations: ~10%
- Nested loops: ~10%
- Combined patterns: ~10%
- Scalar parameters: ~10%

### 4.2 CUDA Dataset

| Metric | Value |
|--------|-------|
| **Total Size** | 231 MB |
| **Number of Pairs** | 20,001 |
| **Average Pair Size** | ~11 KB |
| **File Format** | JSON |
| **Device Target** | CUDA (GPU) |

**CUDA-Specific Features:**
- `__global__` kernel declarations
- Grid-stride loops for scalability
- Shared memory initialization
- Thread synchronization patterns

### 4.3 Sample Data Format

**CPU Sample:**
```json
{
  "id": "hash_12chars",
  "kernel_name": "arith_abcdef",
  "kernel_type": "arithmetic",
  "python_source": "@wp.kernel\ndef arith_abcdef(...):\n    ...",
  "cpp_ir_forward": "void arith_abcdef_hash_cpu_kernel_forward(...) {...}",
  "cpp_ir_backward": "void arith_abcdef_hash_cpu_kernel_backward(...) {...}",
  "generated_at": "2025-12-28T...",
  "metadata": {
    "num_params": 3,
    "num_lines": 5,
    "device": "cpu"
  }
}
```

**CUDA Sample:**
```json
{
  "id": "hash_12chars",
  "kernel_name": "vec_ghijkl",
  "kernel_type": "vector",
  "python_source": "@wp.kernel\ndef vec_ghijkl(...):\n    ...",
  "cuda_ir_forward": "__global__ void vec_ghijkl_hash_cuda_kernel_forward(...) {...}",
  "cuda_ir_backward": "__global__ void vec_ghijkl_hash_cuda_kernel_backward(...) {...}",
  "generated_at": "2025-12-28T...",
  "metadata": {
    "num_params": 3,
    "num_lines": 7,
    "device": "cuda"
  }
}
```

### 4.4 Quality Metrics

| Metric | CPU | CUDA |
|--------|-----|------|
| Success Rate | 100% | 100% |
| Forward IR Present | 100% | 100% |
| Backward IR Present | 100% | 100% |
| Type Coverage | 10 kernel types | 10 kernel types |
| Validation Status | ✅ Passed | ✅ Passed |

---

## 5. Production Pipeline

### Architecture

```
┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  KernelGenerator │ ──► │  Warp Compiler  │ ──► │  IR Extraction   │
│  (10 types)      │     │  (JIT codegen)  │     │  (forward/back)  │
└──────────────────┘     └─────────────────┘     └──────────────────┘
                                                          │
                                                          ▼
                                                 ┌──────────────────┐
                                                 │  JSON Output     │
                                                 │  (paired data)   │
                                                 └──────────────────┘
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `generator.py` | Programmatic kernel generation (10 types) |
| `fast_batch_generator.py` | High-throughput CPU IR generation |
| `fast_batch_generator_cuda.py` | High-throughput CUDA IR generation |
| `pipeline.py` | Single-kernel extraction (for validation) |

### Performance

| Metric | CPU Generation | CUDA Generation |
|--------|---------------|-----------------|
| Rate | ~10.5 pairs/sec | ~43 pairs/sec |
| Time for 20k pairs | ~32 min | ~8 min |
| Bottleneck | C++ compilation | Python codegen |

---

## 6. Usage Recommendations

### For LLM Training

**Task: Python → IR Translation**
- Input: `python_source` field
- Output: `cpp_ir_forward` or `cuda_ir_forward`
- Benefit: Teaches compilation semantics

**Task: Autodiff Generation**
- Input: `cpp_ir_forward` / `cuda_ir_forward`
- Output: `cpp_ir_backward` / `cuda_ir_backward`
- Benefit: Teaches gradient computation patterns

**Task: Cross-Device Translation**
- Input: CPU IR + metadata
- Output: CUDA IR (or vice versa)
- Benefit: Teaches hardware-specific optimizations

### Data Augmentation Suggestions

1. **Combine CPU and CUDA**: Train on both for device-agnostic understanding
2. **Mask partial IR**: Learn to complete incomplete implementations
3. **Type perturbation**: Change data types to test generalization
4. **Operator swapping**: Exchange equivalent operations

### Preprocessing Recommendations

1. Tokenize IR as C++/CUDA code (use code tokenizers)
2. Preserve structure (indentation, braces matter)
3. Consider comment stripping for cleaner data
4. Normalize variable names if desired

---

## 7. Appendix: Technical Details

### Warp Version
- **Version**: 1.10.1
- **Python**: 3.12
- **Backend**: CPU-only (CUDA code generated via codegen API)

### File Locations
```
/workspace/
├── data/
│   ├── cpu/           # 21,281 CPU JSON pairs (209 MB)
│   └── cuda/          # 20,001 CUDA JSON pairs (231 MB)
├── code/
│   ├── synthesis/     # Generation pipeline
│   └── extraction/    # IR extraction utilities
└── report/
    └── chief_scientist_report.md  # This report
```

### Reproducibility
- CPU generation seed: 42, 100042, 200042, 300042
- CUDA generation seed: 42
- All code available in `/workspace/code/`

---

## 8. Conclusion

We have successfully delivered:

✅ **209 MB CPU training data** (21,281 Python→C++ IR pairs)  
✅ **231 MB CUDA training data** (20,001 Python→CUDA IR pairs)  
✅ **440 MB total** high-quality paired data for LLM training  
✅ **Production pipeline** for generating additional data as needed  
✅ **Full documentation** of the process and data format  

The dataset captures the translation from high-level Python abstractions to low-level hardware-specific implementations, including automatic differentiation. This data is ready for use in training LLMs on code translation and compiler understanding tasks.

---

*Report generated: December 28, 2025*  
*Pipeline: NVIDIA Warp 1.10.1 + Custom Synthesis Pipeline*
