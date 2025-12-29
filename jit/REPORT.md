# JIT Code Synthesis Dataset - Technical Report

**For: Chief Scientist**  
**Date: December 2024**

---

## Executive Summary

This project synthesizes paired training data (Python → IR) for Large Language Model training using NVIDIA Warp's JIT compilation framework. We have built a production pipeline that generates CPU C++ code and CUDA GPU code from Python kernel definitions, creating structured training pairs for learning code translation patterns.

---

## 1. Introduction to JIT Compilation

### What is JIT Compilation?

**Just-In-Time (JIT) compilation** is a method of executing computer code that involves compilation during program execution rather than prior to execution. Unlike traditional ahead-of-time (AOT) compilation, JIT compilers translate code at runtime, enabling:

- **Dynamic optimization**: Code can be optimized based on runtime behavior
- **Platform portability**: Same source code runs on different architectures
- **Interactivity**: Rapid development cycles without explicit compilation steps

### Why JIT Matters for ML/Scientific Computing

In machine learning and scientific computing, JIT compilation bridges the gap between:
- **Productivity**: High-level Python code for rapid prototyping
- **Performance**: Low-level optimized code for production workloads

Key benefits:
1. **GPU acceleration without CUDA expertise**: Researchers write Python, JIT generates CUDA
2. **Automatic differentiation**: Backward passes generated automatically for gradient computation
3. **Cross-platform deployment**: Same code runs on CPU, CUDA, and other backends

---

## 2. Intermediate Representation (IR)

### What is IR?

**Intermediate Representation (IR)** is a data structure or code used internally by a compiler to represent source code during translation. IR serves as the bridge between high-level source code and low-level machine code.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Python       │ -> │ IR           │ -> │ Machine Code │
│ Source Code  │    │ (C++/CUDA)   │    │ (Binary)     │
└──────────────┘    └──────────────┘    └──────────────┘
```

### IR in This Dataset

In our dataset, the IR takes the form of generated C++ (for CPU) or CUDA (for GPU) code. This IR:
- Preserves semantic meaning of the original Python
- Adds explicit type information
- Implements automatic differentiation (backward pass)
- Includes platform-specific optimizations

---

## 3. NVIDIA Warp

### Overview

[NVIDIA Warp](https://github.com/NVIDIA/warp) is a Python framework for writing high-performance simulation and graphics code. It provides:

- **@wp.kernel decorator**: Marks Python functions for JIT compilation
- **Differentiable programming**: Automatic gradient computation
- **Multi-backend support**: CPU and CUDA from single source

### How Warp Compiles Python to CPU/GPU Code

```
Python Kernel (@wp.kernel)
        │
        ▼
┌───────────────────┐
│  AST Analysis     │  ← Parse Python source
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Type Inference   │  ← Determine array/scalar types
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Code Generation  │  ← Generate C++/CUDA
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Compilation      │  ← JIT compile to binary
└───────────────────┘
```

### Key Features Used in This Dataset

1. **Kernel Types Generated**:
   - `arithmetic`: Basic math operations (+, -, *, /)
   - `conditional`: If/else branching logic
   - `loop`: For loops with accumulators
   - `math`: Transcendental functions (sin, cos, exp, etc.)
   - `vector`: wp.vec3 operations (physics simulations)
   - `atomic`: Thread-safe accumulation (wp.atomic_add)
   - `nested`: Nested loop patterns
   - `multi_cond`: Multiple elif branches
   - `combined`: Multi-pattern kernels
   - `scalar_param`: Kernels with scalar parameters

2. **Generated Code Structure**:
   - Forward kernel: Computes the primary operation
   - Backward kernel: Computes gradients for automatic differentiation

---

## 4. Dataset Overview

### 4.1 CPU Dataset

| Metric | Value |
|--------|-------|
| **Location** | `jit/data/cpu/` |
| **Target Size** | 200 MB |
| **Final Size** | **201 MB (20,939 pairs)** ✓ COMPLETE |
| **Format** | JSON files |
| **Backend** | C++ (`.cpp`) |

**Kernel Type Distribution**:
- arithmetic, conditional, loop, math, vector
- atomic, nested, multi_cond, combined, scalar_param
- Each type ~10% of dataset

**Note**: CPU code generation runs at ~0.7 pairs/sec because it requires full C++ compilation for each kernel.

### 4.2 CUDA Dataset

| Metric | Value |
|--------|-------|
| **Location** | `jit/data/cuda/` |
| **Target Size** | 200 MB |
| **Final Size** | **226 MB (19,610 pairs)** ✓ COMPLETE |
| **Format** | JSON files |
| **Backend** | CUDA (`.cu`) |

**Note**: CUDA code generation does not require a physical GPU. We use Warp's `codegen("cuda")` function which generates CUDA source without device execution. This is much faster (~20+ pairs/sec) than CPU generation.

---

## 5. Data Format

Each JSON file contains a paired Python→IR sample:

```json
{
  "id": "abc123def456",
  "kernel_name": "arith_example",
  "kernel_type": "arithmetic",
  "python_source": "@wp.kernel\ndef arith_example(a: wp.array(dtype=float), ...):\n    ...",
  "ir_forward": "void arith_example_hash_cpu_kernel_forward(...) { ... }",
  "ir_backward": "void arith_example_hash_cpu_kernel_backward(...) { ... }",
  "device": "cpu",
  "generated_at": "2024-12-28T...",
  "metadata": {
    "num_params": 3,
    "num_lines": 2,
    "module_id": "wp_..."
  }
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `id` | Unique hash of Python source |
| `kernel_name` | Generated kernel function name |
| `kernel_type` | Category (arithmetic, loop, etc.) |
| `python_source` | Original Python kernel code |
| `ir_forward` | Generated forward pass C++/CUDA |
| `ir_backward` | Generated backward pass (gradients) |
| `device` | Target device ("cpu" or "cuda") |
| `metadata` | Additional kernel properties |

---

## 6. Usage Notes

### Loading the Data

```python
import json
from pathlib import Path

# Load all CPU samples
cpu_pairs = []
for f in Path("jit/data/cpu").glob("*.json"):
    with open(f) as fp:
        cpu_pairs.append(json.load(fp))

print(f"Loaded {len(cpu_pairs)} CPU pairs")
```

### Recommended Preprocessing for LLM Training

1. **Tokenization**: Use code-aware tokenizers (CodeLlama, StarCoder)
2. **Pair formatting**: 
   ```
   ### Python Kernel:
   {python_source}
   
   ### Generated IR:
   {ir_forward}
   ```
3. **Data augmentation**: Include both forward and backward passes
4. **Filtering**: Remove pairs where `ir_forward` is empty

### Production Commands

```bash
# Generate CPU data (target: 200MB)
python3 jit/code/synthesis/produce_cpu_data.py --target 200

# Generate CUDA data (no GPU required, target: 200MB)
python3 jit/code/synthesis/produce_cuda_data.py --target 200
```

---

## 7. Technical Notes

### Performance Characteristics

- **CPU generation**: ~0.5-1.0 pairs/second (includes C++ compilation)
- **CUDA generation**: ~20-30 pairs/second (codegen only, no compilation)
- **Average pair size**: ~8-10 KB (JSON with both forward/backward)

### Known Limitations

1. **Compilation cache growth**: Warp's cache grows over time, slowing generation. Mitigated by periodic cache clearing.
2. **GPU-only features**: Some CUDA-specific patterns (shared memory, warp intrinsics) not yet covered.
3. **Complex kernels**: Very large nested structures may fail extraction.

---

## Appendix: File Structure

```
jit/
├── code/
│   ├── extraction/
│   │   └── ir_extractor.py      # Core IR extraction logic
│   └── synthesis/
│       ├── generator.py          # Kernel pattern generator
│       ├── pipeline.py           # End-to-end pipeline
│       ├── produce_cpu_data.py   # CPU production script
│       └── produce_cuda_data.py  # CUDA production script
├── data/
│   ├── cpu/                      # CPU training pairs
│   │   ├── *.json               # Individual pair files
│   │   └── stats.json           # Generation statistics
│   └── cuda/                     # CUDA training pairs
│       ├── *.json
│       └── stats.json
└── REPORT.md                     # This document
```

---

*Report generated by automated dataset production pipeline.*
