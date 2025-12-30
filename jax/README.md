# JAX JIT Code Synthesis Dataset

Training data generation pipeline for LLM code translation: Python → HLO/XLA (with forward and backward passes).

## Overview

This project uses JAX's JIT compilation to generate high-quality Python→HLO training pairs for large language models. Each sample contains:
- Python function source code
- **HLO Forward** code (unoptimized XLA IR)
- **HLO Optimized** code (after XLA optimization passes)
- **HLO Backward** code (gradient computation)

## Dataset

**Location:** `jax/data/training_all.jsonl`  
**Format:** JSONL (one JSON per line)

### Sample Format

```json
{
  "id": 0,
  "kernel_name": "elementwise_qahf",
  "python": "def elementwise_qahf(a, b):\n    return a + b",
  "hlo_forward": "HloModule jit_elementwise_qahf...",
  "hlo_backward": "HloModule jit_backward...",
  "hlo_optimized": "HloModule jit_optimized...",
  "type": "generate_simple_elementwise"
}
```

Each sample includes:
- **`hlo_forward`**: Unoptimized HLO before XLA passes
- **`hlo_backward`**: HLO for gradient computation (autodiff)
- **`hlo_optimized`**: Optimized HLO after fusion and other passes

## Kernel Types (10+ categories)

| Type | Description | Example |
|------|-------------|---------|
| `elementwise` | Basic arithmetic (+, -, *) | `a + b` |
| `scalar_array` | Scalar + array ops | `alpha * x + y` |
| `unary` | Math functions | `jnp.sin(a)` |
| `branch` | Conditionals | `jnp.where(a > 0, ...)` |
| `loop` | Scan/fori_loop | `lax.fori_loop(...)` |
| `reduction` | Sum, mean, max | `jnp.sum(a)` |
| `vector` | Dot product, norm | `jnp.sum(a * b)` |
| `multi_statement` | Chained ops | `temp = a+b; jnp.sqrt(temp)` |
| `nested_branch` | Nested where | Nested conditionals |
| `compound` | Mixed patterns | Complex multi-op functions |
| `matmul` | Matrix multiply | `jnp.matmul(a, b)` |
| `softmax` | Softmax activation | `jax.nn.softmax(x)` |
| `attention` | Scaled dot-product | Transformer attention |
| `layernorm` | Layer normalization | LayerNorm computation |

## Quick Start

### Requirements
```bash
pip install jax jaxlib numpy
# For GPU support:
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Generate Training Data
```bash
cd jax

# Generate 100 pairs (demo)
python3 code/synthesis/pipeline.py --count 100

# Generate to JSONL file with forward + backward
python3 code/synthesis/pipeline.py --count 1000 --output data/my_data.jsonl --jsonl --mode both

# Generate forward-only data
python3 code/synthesis/pipeline.py --count 1000 --output data/forward.jsonl --jsonl --mode forward

# Include extended ML kernels (attention, softmax, etc.)
python3 code/synthesis/pipeline.py --count 1000 --output data/ml_kernels.jsonl --jsonl --extended
```

### Test IR Extraction
```bash
python3 jax/code/extraction/ir_extractor.py
python3 jax/code/extraction/test_ir_extractor.py
```

## Project Structure

```
jax/
├── code/
│   ├── extraction/
│   │   ├── ir_extractor.py       # Core HLO extraction
│   │   └── test_ir_extractor.py  # Tests
│   ├── synthesis/
│   │   ├── generator.py          # 10+ kernel type generators
│   │   └── pipeline.py           # Main synthesis pipeline
│   └── examples/
│       ├── test_add_kernel.py
│       ├── test_dot_product.py
│       └── test_saxpy.py
├── data/
│   ├── training_all.jsonl        # Main dataset
│   └── samples/
├── notes/
│   ├── jax_basics.md             # JAX compilation flow
│   └── hlo_format.md             # HLO structure docs
└── requirements.txt
```

## How It Works

1. **Function Generation**: `generator.py` creates random Python functions from 10+ templates
2. **JIT Lowering**: JAX lowers functions to HLO via `jax.jit(fn).lower(inputs)`
3. **HLO Extraction**: `ir_extractor.py` captures forward, backward, and optimized HLO
4. **Pair Creation**: Pipeline combines Python + HLO into training samples

## Key Features

- **Forward + Backward**: Both computation and gradient HLO included
- **Optimized HLO**: XLA-optimized representation showing fusion patterns
- **Reproducible**: Seeded random generation for reproducibility
- **10+ Kernel Types**: Balanced coverage of common ML patterns
- **Extended ML Kernels**: Attention, softmax, layer norm, etc.
- **Production Ready**: Validated, clean JSONL format

## HLO vs Warp IR

| Aspect | JAX HLO | Warp C++/CUDA |
|--------|---------|---------------|
| Level | High-level ops | Low-level code |
| Format | XLA IR text | C++/CUDA source |
| Fusion | XLA-managed | Manual |
| Portability | Multi-backend | CPU/CUDA specific |
| Gradients | Via `jax.grad` | Via adjoint |

## Example Output

### Python Input
```python
def elementwise_abcd(a, b):
    """Elementwise operation on two arrays."""
    return a + b
```

### HLO Forward
```
HloModule jit_elementwise_abcd, entry_computation_layout={(f32[10],f32[10])->f32[10]}

ENTRY main.4 {
  Arg_0.1 = f32[10] parameter(0)
  Arg_1.2 = f32[10] parameter(1)
  ROOT add.3 = f32[10] add(Arg_0.1, Arg_1.2)
}
```

### HLO Backward (Gradient)
```
HloModule jit_grad_fn, entry_computation_layout={(f32[10],f32[10])->(f32[10],f32[10])}

ENTRY main.6 {
  Arg_0.1 = f32[10] parameter(0)
  Arg_1.2 = f32[10] parameter(1)
  constant.3 = f32[] constant(1)
  broadcast.4 = f32[10] broadcast(constant.3), dimensions={}
  ROOT tuple.5 = (f32[10], f32[10]) tuple(broadcast.4, broadcast.4)
}
```

## License

JAX is licensed under Apache 2.0.
