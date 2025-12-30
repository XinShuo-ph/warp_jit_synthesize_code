# JAX JIT Code Synthesis Dataset (Migrated from Warp)

Training data generation pipeline for LLM code translation: Python (JAX) → HLO (StableHLO).

## Overview

This project uses JAX's JIT compilation to generate high-quality Python→HLO training pairs for large language models. Each sample contains:
- Python kernel source code (using `jax.jit` and `jax.numpy`)
- **HLO code** (StableHLO IR text)

## Dataset

**Location:** `jit/data/training_all.jsonl`  
**Size:** Generated on demand
**Format:** JSONL (one JSON per line)

### Sample Format

```json
{
  "id": 0,
  "kernel_name": "scalar_arr_qahf",
  "python": "@jax.jit\ndef scalar_arr_qahf(...):\n    ...",
  "cpp": "module @jit_scalar_arr_qahf ...",
  "cuda": "module @jit_scalar_arr_qahf ...",
  "type": "generate_scalar_array_op"
}
```

Each sample includes:
- **`cpp`**: HLO code (StableHLO)
- **`cuda`**: Same HLO code (or optimized if backend available)

## Kernel Types (10 categories)

| Type | Description | Example |
|------|-------------|---------|
| `elementwise` | Basic arithmetic (+, -, *) | `c = a + b` |
| `scalar_array` | Scalar + array ops | `out = alpha * x + y` |
| `unary` | Math functions | `b = jnp.sin(a)` |
| `branch` | Conditionals | `jnp.where(...)` |
| `loop` | Loops (unrolled or vectorized) | `a * n` |
| `reduction` | Reductions | `jnp.sum(a)` |
| `vector` | Vector operations | `jnp.dot(a, b)` |
| `multi_statement` | Chained ops | `temp = a+b; c = sqrt(temp)` |
| `nested_branch` | Nested conditionals | `jnp.where(cond1, ...)` |
| `compound` | Mixed patterns | Complex multi-op kernels |

## Quick Start

### Requirements
```bash
pip install jax jaxlib numpy
```

### Generate Training Data
```bash
cd jit

# Generate 100 pairs (demo)
python3 code/synthesis/pipeline.py --count 100

# Generate to JSONL file
python3 code/synthesis/pipeline.py --count 1000 --output data/jax_data.jsonl --jsonl --device both
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
│   │   └── ir_extractor.py      # Core IR extraction (JAX HLO)
│   └── synthesis/
│       ├── generator.py         # 10 kernel type generators (JAX)
│       ├── pipeline.py          # Main synthesis pipeline
│       └── batch_generator.py   # Scalable batch generation
├── data/
│   └── samples/                 # Sample pairs (JSON)
└── notes/
```

## How It Works

1. **Kernel Generation**: `generator.py` creates random JAX kernels from 10 templates
2. **JIT Compilation**: JAX compiles kernels to HLO
3. **IR Extraction**: `ir_extractor.py` captures the generated HLO text
4. **Pair Creation**: Pipeline combines Python + HLO into training samples
