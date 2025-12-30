# JAX JIT Code Synthesis Dataset

Training data generation pipeline for LLM code translation: Python → HLO/XLA IR (with forward and backward passes).

## Overview

This project uses JAX's JIT compilation to generate high-quality Python→HLO training pairs for large language models. Each sample contains:
- Python function source code
- **Jaxpr** (JAX intermediate representation)
- **HLO** (High-Level Optimizer IR from XLA)
- **StableHLO** (MLIR-based representation, when available)

## Dataset

**Location:** `jit/data/training_all.jsonl`  
**Size:** 1,500 training pairs (~6MB)  
**Format:** JSONL (one JSON per line)

### Sample Format

```json
{
  "id": 0,
  "function_name": "elementwise_qahf",
  "python": "def elementwise_qahf(a, b):\n    return a + b",
  "jaxpr": "{ lambda ; a:f32[64] b:f32[64]. let c:f32[64] = add a b in (c,) }",
  "hlo": "HloModule jit_elementwise...\nENTRY main {\n  ...\n}",
  "type": "generate_simple_elementwise"
}
```

Each sample includes:
- **`jaxpr`**: JAX's traced intermediate representation with forward + backward
- **`hlo`**: XLA's HLO IR with forward + backward functions
- **`stablehlo`**: MLIR StableHLO representation (when available)

## Function Types (10 categories)

| Type | Description | Example |
|------|-------------|---------|
| `elementwise` | Basic arithmetic (+, -, *) | `return a + b` |
| `scalar_array` | Scalar + array ops | `return alpha * x + y` |
| `unary` | Math functions | `return jnp.sin(a)` |
| `branch` | Conditionals (jnp.where) | `jnp.where(a > 0, ...)` |
| `reduction` | Sum, mean, max, min | `jnp.sum(a)` |
| `dot_product` | Vector dot product | `jnp.dot(a, b)` |
| `matmul` | Matrix multiplication | `jnp.matmul(a, b)` |
| `multi_statement` | Chained ops | `temp = a+b; jnp.sqrt(temp)` |
| `nested_branch` | Nested conditionals | Multiple `jnp.where` |
| `compound` | Mixed patterns | Complex multi-op functions |

## Quick Start

### Requirements
```bash
pip install jax jaxlib numpy
```

### Generate Training Data
```bash
cd jit

# Generate 100 pairs with jaxpr + HLO (demo)
python3 code/synthesis/pipeline.py --count 100

# Generate to JSONL file with both jaxpr and HLO
python3 code/synthesis/pipeline.py --count 1000 --output data/my_data.jsonl --jsonl --ir-type both

# Generate jaxpr-only data
python3 code/synthesis/pipeline.py --count 1000 --output data/jaxpr_only.jsonl --jsonl --ir-type jaxpr

# Generate HLO-only data
python3 code/synthesis/pipeline.py --count 1000 --output data/hlo_only.jsonl --jsonl --ir-type hlo
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
│   │   └── ir_extractor.py      # Core IR extraction (jaxpr + HLO)
│   └── synthesis/
│       ├── generator.py         # 10 function type generators
│       ├── pipeline.py          # Main synthesis pipeline
│       └── batch_generator.py   # Scalable batch generation
├── data/
│   ├── training_all.jsonl       # Main dataset
│   └── samples/                 # Sample pairs (JSON)
└── notes/
    ├── ir_format.md             # JAX IR structure docs
    └── warp_basics.md           # Legacy Warp notes
```

## How It Works

1. **Function Generation**: `generator.py` creates random Python functions from 10 templates
2. **JIT Compilation**: JAX traces and compiles functions via XLA
3. **IR Extraction**: `ir_extractor.py` captures jaxpr, HLO, and StableHLO
4. **Pair Creation**: Pipeline combines Python + IR into training samples

## Key Features

- **Multiple IR Formats**: jaxpr, HLO, and StableHLO included
- **Forward + Backward**: Both gradient functions included (via `jax.grad`)
- **Reproducible**: Seeded random generation for reproducibility
- **10 Function Types**: Balanced coverage of common patterns
- **Production Ready**: Validated, clean JSONL format

## JAX IR Formats Explained

### Jaxpr (JAX Expression)
- High-level functional representation
- Shows primitive operations and data flow
- Easy to read and understand

```
{ lambda ; a:f32[64] b:f32[64]. let
    c:f32[64] = add a b
  in (c,) }
```

### HLO (High-Level Optimizer)
- XLA's intermediate representation
- Lower-level, closer to hardware
- Shows memory layout and scheduling

```
HloModule jit_func, entry_computation_layout={(f32[64]{0}, f32[64]{0})->f32[64]{0}}
ENTRY main {
  p0 = f32[64]{0} parameter(0)
  p1 = f32[64]{0} parameter(1)
  ROOT add = f32[64]{0} add(p0, p1)
}
```

### StableHLO (MLIR-based)
- Portable, versioned HLO dialect
- MLIR infrastructure compatibility
- Better for cross-platform targets

## License

Uses JAX (Apache 2.0 license).
