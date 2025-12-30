# JAX JIT Code Synthesis Dataset

Training data generation pipeline for LLM code translation: Python → compiler IR (CPU HLO + optional GPU HLO, with forward and backward passes).

## Overview

This project uses JAX's `jit` compilation pipeline to generate high-quality Python→IR training pairs for large language models. Each sample contains:
- Python function source code
- **CPU HLO** (forward + backward)
- **GPU HLO** (forward + backward), if a GPU backend is available

## Dataset

**Location:** `jit/data/training_all.jsonl`  
**Size:** 1,500 training pairs (18MB)  
**Format:** JSONL (one JSON per line)

### Sample Format

```json
{
  "id": 0,
  "kernel_name": "scalar_arr_qahf",
  "python": "def scalar_arr_qahf(...):\n    ...",
  "cpp": "### FORWARD (CPU HLO)\n...\n### BACKWARD (CPU HLO)\n...",
  "cuda": "### FORWARD (GPU HLO)\n...\n### BACKWARD (GPU HLO)\n...",
  "type": "generate_scalar_array_op"
}
```

Each sample includes:
- **`cpp`**: CPU compiler IR (HLO text) including forward + backward sections
- **`cuda`**: GPU compiler IR (HLO text) including forward + backward sections (optional)

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
pip install -r requirements.txt
```

### Generate Training Data
```bash
cd jit

# Generate 100 pairs with CPU IR and (if available) GPU IR
python3 code/synthesis/pipeline.py --count 100

# Generate to JSONL file with CPU IR and (if available) GPU IR
python3 code/synthesis/pipeline.py --count 1000 --output data/my_data.jsonl --jsonl --device both

# Generate CPU-only data
python3 code/synthesis/pipeline.py --count 1000 --output data/cpu_only.jsonl --jsonl --device cpu

# Generate GPU-only data (skips samples if GPU backend not available)
python3 code/synthesis/pipeline.py --count 1000 --output data/cuda_only.jsonl --jsonl --device cuda
```

### Test IR Extraction
```bash
python3 jit/code/extraction/ir_extractor.py
```

## Project Structure

```
jit/
├── code/
│   ├── extraction/
│   │   └── ir_extractor.py      # Core IR extraction (CPU + CUDA)
│   └── synthesis/
│       ├── generator.py         # 10 kernel type generators
│       ├── pipeline.py          # Main synthesis pipeline
│       └── batch_generator.py   # Scalable batch generation
├── data/
│   ├── training_all.jsonl       # Main dataset (1,500 pairs, 18MB)
│   └── samples/                 # Sample pairs (JSON)
└── notes/
    ├── warp_basics.md           # Warp compilation flow
    └── ir_format.md             # C++ IR structure docs
```

## How It Works

1. **Kernel Generation**: `generator.py` creates random JAX functions from 10 templates
2. **JIT Compilation**: JAX lowers functions to compiler IR (HLO) for CPU (and GPU if available)
3. **IR Extraction**: `ir_extractor.py` captures forward and backward compiler IR
4. **Pair Creation**: Pipeline combines Python + IR into training samples

## Key Features

- **CPU + CUDA**: Both backends included in every sample
- **Forward + Backward**: Both gradient functions included (critical for differentiable programming)
- **Reproducible**: Seeded random generation for reproducibility
- **10 Kernel Types**: Balanced coverage of common GPU patterns
- **Production Ready**: Validated, clean JSONL format

## CPU vs CUDA Code Differences

**CPU code** uses:
- Sequential execution via `for (task_index = 0; ...)`
- Args passed via struct pointer

**CUDA code** uses:
- Parallel execution via `blockIdx`, `threadIdx`
- Grid-stride loop pattern
- Direct parameter passing

## License

Uses JAX (Apache-2.0 license).
