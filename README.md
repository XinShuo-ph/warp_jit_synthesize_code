# JAX JIT Code Synthesis Dataset

Training data generation pipeline for LLM code translation: Python → HLO/XLA (with forward and backward passes).

## Overview

This project uses JAX's JIT compilation to generate high-quality Python→HLO training pairs for large language models. Each sample contains:
- Python function source code
- **JAXPR** (JAX's intermediate representation)
- **HLO code** (High-Level Optimizer IR) for CPU/GPU execution

## Dataset

**Location:** `jit/data/training_all.jsonl`  
**Size:** Training pairs generated with JAX  
**Format:** JSONL (one JSON per line)

### Sample Format

```json
{
  "id": 0,
  "function_name": "scalar_arr_qahf",
  "python": "def scalar_arr_qahf(...):\n    ...",
  "jaxpr": "{ lambda ; a:f32[4] b:f32[4]. let ... }",
  "hlo": "HloModule jit_scalar_arr_qahf\n...",
  "type": "generate_scalar_array_op"
}
```

Each sample includes:
- **`jaxpr`**: JAX intermediate representation
- **`hlo`**: XLA HLO code (optimized for CPU/GPU execution)

## Function Types (10 categories)

| Type | Description | Example |
|------|-------------|---------|
| `elementwise` | Basic arithmetic (+, -, *) | `return a + b` |
| `scalar_array` | Scalar + array ops | `return alpha * x + y` |
| `unary` | Math functions | `return jnp.sin(a)` |
| `branch` | Conditionals | `return jnp.where(a > 0, ...)` |
| `loop` | Scan operations | `jax.lax.scan(...)` |
| `reduction` | Sum, mean, etc. | `return jnp.sum(a * b)` |
| `vector` | Vector operations | `return jnp.linalg.norm(a)` |
| `multi_statement` | Chained ops | `temp = a+b; return sqrt(temp)` |
| `nested_branch` | Nested conditionals | `jnp.where(..., jnp.where(...))` |
| `compound` | Mixed patterns | Complex multi-op functions |

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
```

### Generate Training Data
```bash
cd jit

# Generate 100 pairs (demo)
python3 code/synthesis/pipeline.py --count 100

# Generate to JSONL file
python3 code/synthesis/pipeline.py --count 1000 --output data/my_data.jsonl --jsonl

# Generate CPU-optimized data
python3 code/synthesis/pipeline.py --count 1000 --output data/cpu_data.jsonl --jsonl --device cpu
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
│   │   └── ir_extractor.py      # Core IR extraction (JAXPR + HLO)
│   └── synthesis/
│       ├── generator.py         # 10 function type generators
│       ├── pipeline.py          # Main synthesis pipeline
│       └── batch_generator.py   # Scalable batch generation
├── data/
│   ├── training_all.jsonl       # Main dataset
│   └── samples/                 # Sample pairs (JSON)
└── notes/
    ├── warp_basics.md           # (Legacy) Warp compilation flow
    └── ir_format.md             # IR structure docs
```

## How It Works

1. **Function Generation**: `generator.py` creates random Python functions from 10 templates
2. **JIT Compilation**: JAX compiles functions using XLA
3. **IR Extraction**: `ir_extractor.py` captures JAXPR and HLO representations
4. **Pair Creation**: Pipeline combines Python + JAXPR + HLO into training samples

## Key Features

- **JAXPR + HLO**: Both intermediate and optimized representations included
- **Forward + Backward**: Gradient functions automatically derived by JAX
- **Reproducible**: Seeded random generation for reproducibility
- **10 Function Types**: Balanced coverage of common patterns
- **Production Ready**: Validated, clean JSONL format
- **Device Agnostic**: JAX automatically optimizes for CPU/GPU/TPU

## JAX vs Warp

This project has been migrated from NVIDIA Warp to JAX:

**JAX advantages**:
- Automatic differentiation (grad, value_and_grad)
- XLA compilation for multiple backends (CPU, GPU, TPU)
- NumPy-compatible API
- JAXPR provides interpretable IR
- HLO provides optimized low-level representation

## License

Uses JAX (Apache 2.0 license).
