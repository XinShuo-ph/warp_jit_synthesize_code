# JAX JIT Code Synthesis Dataset

Training data generation pipeline for LLM code translation: Python → HLO/XLA (with forward and backward passes).

## Overview

This project uses JAX's JIT compilation to generate high-quality Python→HLO training pairs for large language models. Each sample contains:
- Python function source code
- **HLO (High Level Optimizer)** representation with forward and backward functions
- **Jaxpr** (JAX's intermediate representation) with forward and backward
- **Optimized HLO** after XLA compilation (optional)

## Dataset

**Location:** `jit/data/training_all.jsonl`  
**Format:** JSONL (one JSON per line)

### Sample Format

```json
{
  "id": 0,
  "kernel_name": "scalar_arr_qahf",
  "python": "def scalar_arr_qahf(alpha, x, y):\n    return alpha * x + y",
  "hlo": "HloModule ... { ... entry_computation ... }",
  "jaxpr": "{ lambda ; a:f32[100] b:f32[100]. let c = add a b in (c,) }",
  "optimized_hlo": "HloModule jit_func, ... optimized ...",
  "type": "generate_scalar_array_op"
}
```

Each sample includes:
- **`hlo`**: HLO (High Level Optimizer) text with forward + backward functions
- **`jaxpr`**: JAX's intermediate representation with forward + backward
- **`optimized_hlo`**: XLA-compiled optimized HLO (when available)

## Kernel Types (11 categories)

| Type | Description | Example |
|------|-------------|---------|
| `elementwise` | Basic arithmetic (+, -, *) | `return a + b` |
| `scalar_array` | Scalar + array ops | `return alpha * x` |
| `unary` | Math functions | `return jnp.sin(a)` |
| `branch` | Conditionals (jnp.where) | `jnp.where(val > 0, ...)` |
| `loop` | For loops (jax.lax.fori_loop) | `fori_loop(0, n, ...)` |
| `reduction` | Reduction ops | `jnp.sum(a)` |
| `vector` | Vector operations | `jnp.sum(a * b, axis=-1)` |
| `multi_statement` | Chained ops | `temp = a+b; return sqrt(temp)` |
| `nested_branch` | Nested conditionals | `jnp.where(a > 0, jnp.where(...))` |
| `compound` | Mixed patterns | Complex multi-op functions |
| `matmul` | Matrix multiplication | `jnp.matmul(a, b)` |

## Quick Start

### Requirements
```bash
pip install jax jaxlib numpy
# Or with GPU support:
pip install jax[cuda12] numpy
```

### Generate Training Data
```bash
cd jit

# Generate 100 pairs with all IR types (demo)
python3 code/synthesis/pipeline.py --count 100

# Generate to JSONL file with HLO and Jaxpr
python3 code/synthesis/pipeline.py --count 1000 --output data/my_data.jsonl --jsonl --ir-type both

# Generate HLO-only data
python3 code/synthesis/pipeline.py --count 1000 --output data/hlo_only.jsonl --jsonl --ir-type hlo

# Generate Jaxpr-only data
python3 code/synthesis/pipeline.py --count 1000 --output data/jaxpr_only.jsonl --jsonl --ir-type jaxpr
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
│   │   └── ir_extractor.py      # Core IR extraction (HLO + Jaxpr)
│   └── synthesis/
│       ├── generator.py         # 11 function type generators
│       ├── pipeline.py          # Main synthesis pipeline
│       └── batch_generator.py   # Scalable batch generation
├── data/
│   ├── training_all.jsonl       # Main dataset
│   └── samples/                 # Sample pairs (JSON)
└── notes/
    ├── warp_basics.md           # Original Warp docs (legacy)
    └── ir_format.md             # IR structure docs
```

## How It Works

1. **Function Generation**: `generator.py` creates random Python functions from 11 templates
2. **JIT Compilation**: JAX compiles functions using XLA
3. **IR Extraction**: `ir_extractor.py` captures the generated HLO and Jaxpr
4. **Pair Creation**: Pipeline combines Python + HLO + Jaxpr into training samples

## Key Features

- **Multiple IR Types**: HLO, Jaxpr, and Optimized HLO included
- **Forward + Backward**: Both gradient functions included (critical for differentiable programming)
- **Reproducible**: Seeded random generation for reproducibility
- **11 Function Types**: Balanced coverage of common numerical patterns
- **Production Ready**: Validated, clean JSONL format

## HLO vs Jaxpr

**Jaxpr** (JAX Program Representation):
- High-level functional representation
- Shows JAX primitive operations (add, mul, etc.)
- Platform-independent

**HLO** (High Level Optimizer):
- XLA's intermediate representation
- Shows the computation graph for XLA compilation
- Platform-specific optimizations available

**Optimized HLO**:
- After XLA compilation optimizations
- Contains platform-specific fusions and optimizations

## Why Forward + Backward Matters

Both forward and backward passes are included because:

1. **Differentiable Programming**: Modern ML requires gradients
2. **Complete Translation Task**: LLM learns full autodiff patterns
3. **Real-World Utility**: Matches actual JAX/XLA compiler output

## License

Uses JAX (Apache 2.0 license).
