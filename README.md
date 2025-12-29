# Warp JIT Code Synthesis Dataset

Training data generation pipeline for LLM code translation: Python → C++ (with forward and backward passes).

## Overview

This project uses NVIDIA Warp's JIT compilation to generate high-quality Python→C++ training pairs for large language models. Each sample contains:
- Python kernel source code
- **Both forward and backward (gradient) C++ functions** in a single field

## Dataset

**Location:** `jit/data/training_all.jsonl`  
**Size:** 1,505 training pairs (8.5MB)  
**Format:** JSONL (one JSON per line)

### Sample Format

```json
{
  "id": 0,
  "kernel_name": "vec_lhwu",
  "python": "@wp.kernel\ndef vec_lhwu(a: wp.array(dtype=wp.vec3), ...):\n    ...",
  "cpp": "... _cpu_kernel_forward(...) {...}\n... _cpu_kernel_backward(...) {...}",
  "type": "generate_vector_kernel"
}
```

The `cpp` field contains:
- **Struct definition**: `wp_args_{kernel_name}` with typed parameters
- **Forward pass**: `{name}_cpu_kernel_forward()` - computes primal values
- **Backward pass**: `{name}_cpu_kernel_backward()` - computes gradients (autodiff)
- **Entry points**: C-exported wrappers for Python FFI

## Kernel Types (10 categories)

| Type | Description | Example |
|------|-------------|---------|
| `elementwise` | Basic arithmetic (+, -, *) | `c[i] = a[i] + b[i]` |
| `scalar_array` | Scalar + array ops | `out[i] = alpha * x[i]` |
| `unary` | Math functions | `b[i] = wp.sin(a[i])` |
| `branch` | Conditionals | `if val > 0: ...` |
| `loop` | For loops | `for i in range(n): ...` |
| `reduction` | Atomic ops | `wp.atomic_add(result, 0, a[i])` |
| `vector` | Vec3 operations | `c[i] = wp.dot(a[i], b[i])` |
| `multi_statement` | Chained ops | `temp = a+b; c = sqrt(temp)` |
| `nested_branch` | Nested if/else | `if a > 0: if a > 1: ...` |
| `compound` | Mixed patterns | Complex multi-op kernels |

## Quick Start

### Requirements
```bash
pip install warp-lang
```

### Generate Training Data
```bash
cd jit

# Generate 100 pairs (demo)
python3 code/synthesis/pipeline.py --count 100

# Generate to JSONL file
python3 code/synthesis/pipeline.py --count 1000 --output data/my_data.jsonl --jsonl

# Batch generation with checkpointing
python3 code/synthesis/batch_generator.py --count 5000 --output data/large.jsonl
```

### Test IR Extraction
```bash
# Run extraction test
python3 jit/code/extraction/ir_extractor.py
```

## Project Structure

```
jit/
├── code/
│   ├── extraction/
│   │   └── ir_extractor.py      # Core IR extraction from Warp
│   └── synthesis/
│       ├── generator.py         # 10 kernel type generators
│       ├── pipeline.py          # Main synthesis pipeline
│       └── batch_generator.py   # Scalable batch generation
├── data/
│   ├── training_all.jsonl       # Main dataset (1,505 pairs)
│   └── samples/                 # Sample pairs (JSON)
└── notes/
    ├── warp_basics.md           # Warp compilation flow
    └── ir_format.md             # C++ IR structure docs
```

## How It Works

1. **Kernel Generation**: `generator.py` creates random Python kernels from 10 templates
2. **JIT Compilation**: Warp compiles kernels and generates C++ code
3. **IR Extraction**: `ir_extractor.py` captures the generated forward/backward C++
4. **Pair Creation**: Pipeline combines Python source + C++ IR into training samples

## Key Features

- **Forward + Backward**: Both gradient functions included (critical for differentiable programming)
- **Reproducible**: Seeded random generation for reproducibility
- **10 Kernel Types**: Balanced coverage of common GPU patterns
- **Production Ready**: Validated, clean JSON format

## Technical Details

The C++ IR includes:
- Type-annotated variables (`wp::float32`, `wp::vec_t<3>`, etc.)
- Explicit memory operations (`wp::load`, `wp::store`, `wp::address`)
- Comments mapping C++ back to Python source lines
- Full autodiff support via adjoint functions

## License

Uses NVIDIA Warp (BSD-3-Clause license).
