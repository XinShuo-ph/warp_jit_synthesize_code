# Warp JIT Code Synthesis - Merged Codebase

## Overview
Python→IR training data synthesis for NVIDIA Warp JIT compiler. Generates pairs of Python kernel source code and corresponding C++ intermediate representation for LLM training.

**Merged from 16 parallel development branches** - combining the best components from each.

## Features
- **10 kernel types**: arithmetic, vector, matrix, control_flow, math, atomic, nested_loop, multi_conditional, combined, scalar_param
- **IR extraction**: Extracts C++ code from compiled Warp kernels
- **Synthesis pipeline**: End-to-end Python→IR pair generation
- **Batch generation**: Scalable data generation with checkpointing
- **FEM examples**: Poisson solver using warp.fem

## Requirements

```bash
pip install warp-lang
```

## Quick Start

```bash
# Generate 10 training samples
python3 code/synthesis/pipeline.py -n 10 -o data/samples

# Generate specific kernel types
python3 code/synthesis/pipeline.py -n 20 -c arithmetic vector matrix

# Test IR extraction
python3 code/extraction/ir_extractor.py

# Run Poisson solver
python3 code/examples/test_poisson.py
```

See [QUICKSTART.md](QUICKSTART.md) for detailed usage.

## File Structure

```
workspace/
├── code/
│   ├── extraction/           # IR extraction
│   │   ├── ir_extractor.py   # Main extraction utility
│   │   └── test_ir_extractor.py
│   ├── synthesis/            # Kernel generation
│   │   ├── generator.py      # 10 kernel type generators
│   │   ├── pipeline.py       # End-to-end pipeline
│   │   └── batch_generator.py
│   └── examples/             # Example kernels
│       ├── poisson_solver.py # FEM Poisson solver
│       └── test_poisson.py
├── data/                     # Generated samples
├── notes/                    # Technical documentation
├── merge_notes/              # Branch analysis notes
└── README.md
```

## Kernel Types

| Category | Description |
|----------|-------------|
| `arithmetic` | Binary/unary arithmetic operations |
| `vector` | vec2/vec3/vec4 operations (dot, cross, normalize) |
| `matrix` | mat22/mat33/mat44 operations (multiply, transpose) |
| `control_flow` | Conditionals and simple loops |
| `math` | Math functions (sin, cos, exp, sqrt, etc.) |
| `atomic` | Atomic operations (add, min, max) |
| `nested_loop` | Nested for loops |
| `multi_conditional` | Multiple elif branches |
| `combined` | Loop + conditional + math combinations |
| `scalar_param` | Kernels with scalar parameters |

## Generated Data Format

```json
{
  "python_source": "@wp.kernel\ndef add_kernel(a, b, c):\n    tid = wp.tid()\n    c[tid] = a[tid] + b[tid]\n",
  "cpp_forward": "void add_kernel_cpu_kernel_forward(...) { ... }",
  "metadata": {
    "kernel_name": "add_kernel",
    "category": "arithmetic",
    "description": "Arithmetic kernel with 2 operations",
    "device": "cpu"
  }
}
```

## Merge Sources

| Component | Source Branch |
|-----------|--------------|
| Base pipeline | 12c4 (10,727 pairs) |
| Extra kernel types | 9177 (nested, multi_cond, combined, scalar_param) |
| Documentation | 82cf, aa30 |
| IR extraction | 12c4 |

---

**Status**: Production Ready  
**Date**: December 28, 2025
