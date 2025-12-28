# Quick Start Guide

## Installation

```bash
pip install warp-lang
```

## Generate Training Data

### Command Line

```bash
# Generate 10 samples
python3 code/synthesis/pipeline.py -n 10 -o data/samples

# Generate 100 samples with specific categories
python3 code/synthesis/pipeline.py -n 100 -o data/batch \
    -c arithmetic vector matrix control_flow

# All 10 kernel types
python3 code/synthesis/pipeline.py -n 100 -o data/full \
    -c arithmetic vector matrix control_flow math atomic \
       nested_loop multi_conditional combined scalar_param
```

### Python API

```python
import sys
sys.path.insert(0, '/workspace/code/synthesis')

from generator import generate_kernel, GENERATORS
from pipeline import synthesize_pair

# List all kernel types
print("Available types:", list(GENERATORS.keys()))

# Generate a specific kernel type
spec = generate_kernel("arithmetic", seed=42)
print(spec.source)

# Synthesize Python→IR pair
pair = synthesize_pair(spec)
print("Python:", pair["python_source"])
print("C++:", pair["cpp_forward"][:200], "...")
```

## Kernel Types

| Type | Description | Example |
|------|-------------|---------|
| `arithmetic` | Binary/unary ops | `c[i] = a[i] + b[i] * 2.0` |
| `vector` | vec3 operations | `wp.dot(a[i], b[i])` |
| `matrix` | Matrix multiply | `m[i] * v[i]` |
| `control_flow` | If/else, loops | `if x > 0: ...` |
| `math` | Math functions | `wp.sin(wp.cos(x))` |
| `atomic` | Atomic ops | `wp.atomic_add(...)` |
| `nested_loop` | Nested loops | `for i: for j: ...` |
| `multi_conditional` | Multiple elif | `if/elif/elif/else` |
| `combined` | Loop+cond+math | Complex patterns |
| `scalar_param` | Scalar args | `x * scale + offset` |

## Run Examples

```bash
# Poisson FEM solver
python3 code/examples/poisson_solver.py
python3 code/examples/test_poisson.py

# IR extraction demo
python3 code/extraction/ir_extractor.py
```

## Output Format

Each sample is saved as JSON:

```json
{
  "python_source": "@wp.kernel\ndef my_kernel(...):\n    ...",
  "cpp_forward": "void my_kernel_cpu_kernel_forward(...) { ... }",
  "metadata": {
    "kernel_name": "my_kernel",
    "category": "arithmetic",
    "description": "...",
    "device": "cpu"
  }
}
```

## Performance

- ~1.2 seconds per kernel compilation
- 100 samples ≈ 2 minutes
- For large batches, use `batch_generator.py` with checkpointing

## Key Files

| File | Purpose |
|------|---------|
| `code/synthesis/generator.py` | 10 kernel generators |
| `code/synthesis/pipeline.py` | Synthesis pipeline |
| `code/extraction/ir_extractor.py` | C++ code extraction |
| `code/examples/poisson_solver.py` | FEM example |
