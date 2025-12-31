# JIT Code Synthesis Dataset

Training data generation pipeline for LLM code translation using JIT compilation.

## Overview

This project generates high-quality training pairs for LLM code translation using two backends:

| Backend | Input | Output | Use Case |
|---------|-------|--------|----------|
| **Warp** | Python kernels | C++/CUDA code | Low-level GPU kernel implementation |
| **JAX** | Python functions | HLO/XLA IR | Compiler optimization patterns |

## Datasets

### Warp Dataset (Python → C++/CUDA)

**Location:** `jit/data/training_all.jsonl`  
**Size:** 1,500 training pairs (18MB)

```json
{
  "id": 0,
  "kernel_name": "scalar_arr_qahf",
  "python": "@wp.kernel\ndef scalar_arr_qahf(...):\n    ...",
  "cpp": "... _cpu_kernel_forward(...) {...}",
  "cuda": "... _cuda_kernel_forward(...) {...}",
  "type": "generate_scalar_array_op"
}
```

### JAX Dataset (Python → HLO/XLA)

**Location:** `jax/data/training_all.jsonl`  
**Size:** 1,500 training pairs (7.2MB)

```json
{
  "id": 0,
  "kernel_name": "elementwise_abcd",
  "python": "def elementwise_abcd(a, b):\n    return a + b",
  "hlo_forward": "HloModule jit_elementwise_abcd...",
  "hlo_backward": "HloModule jit_grad_fn...",
  "hlo_optimized": "HloModule optimized...",
  "type": "generate_simple_elementwise"
}
```

## Quick Start

### Warp (C++/CUDA)
```bash
pip install warp-lang
cd jit
python3 code/synthesis/pipeline.py --count 100 --output data/my_data.jsonl --jsonl
```

### JAX (HLO/XLA)
```bash
pip install jax jaxlib
cd jax
python3 code/synthesis/pipeline.py --count 100 --output data/my_data.jsonl --jsonl
```

## Kernel Types

Both backends support 10 kernel types:

| Type | Description | Example |
|------|-------------|---------|
| Elementwise | Basic arithmetic | `a + b` |
| Scalar-Array | Scalar ops | `alpha * x + y` |
| Unary | Math functions | `sin(a)` |
| Branch | Conditionals | `if a > 0: ...` |
| Loop | Iterations | `for i in range(n): ...` |
| Reduction | Aggregations | `sum(a)` |
| Vector | Dot/norm | `dot(a, b)` |
| Multi-Statement | Chained ops | `temp = a+b; sqrt(temp)` |
| Nested Branch | Nested if | `if a: if b: ...` |
| Compound | Mixed patterns | Complex multi-op |

JAX also includes extended ML kernels: matmul, softmax, attention, layernorm, etc.

## Project Structure

```
.
├── jit/                    # Warp backend (Python → C++/CUDA)
│   ├── code/
│   │   ├── extraction/     # IR extraction
│   │   └── synthesis/      # Kernel generation
│   └── data/               # Training data
├── jax/                    # JAX backend (Python → HLO)
│   ├── code/
│   │   ├── extraction/     # HLO extraction
│   │   └── synthesis/      # Function generation
│   └── data/               # Training data
├── README.md
└── REPORT.md
```

## Key Features

- **Forward + Backward**: Both computation and gradient code included
- **Multiple Backends**: Warp (C++/CUDA) and JAX (HLO/XLA)
- **Reproducible**: Seeded random generation
- **10+ Kernel Types**: Balanced coverage of common patterns
- **Production Ready**: Validated JSONL format

## License

- Warp: BSD-3-Clause (NVIDIA)
- JAX: Apache 2.0 (Google)
