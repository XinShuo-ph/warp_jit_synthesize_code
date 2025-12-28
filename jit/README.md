# Warp JIT Code Synthesis for LLM Training Data

A comprehensive system for generating Python→IR (Intermediate Representation) training pairs using NVIDIA's Warp package. This project synthesizes diverse kernel code and extracts compiled C++ representations for machine learning applications.

## Overview

This merged codebase combines the best work from 16 parallel development branches, providing:

- **10 kernel type generators** (arithmetic, vector, matrix, control_flow, math, atomic, nested_loop, multi_conditional, scalar_param, loop)
- **Complete IR extraction pipeline** (Python source → compiled C++ code)
- **Batch generation system** (~180-380 pairs/second)
- **175+ sample data pairs** included
- **Comprehensive test suites** (categorized test cases, fixture kernels)
- **Analysis and validation tools**

## Quick Start

See [QUICKSTART.md](./QUICKSTART.md) for detailed setup instructions.

### Installation

```bash
pip install warp-lang
```

### Generate Your First Samples

```bash
# Test IR extraction with diverse kernel types
python3 code/extraction/test_ir_extractor.py

# Generate 10 Python→IR pairs
python3 code/synthesis/pipeline.py -n 10 -o data/test

# Generate large batch (1000 samples)
python3 code/synthesis/batch_generator.py --count 1000 --output data/custom
```

### Validate Poisson Solver (FEM)

```bash
python3 code/examples/test_poisson.py
```

## Project Structure

```
jit/
├── code/
│   ├── extraction/           # IR extraction from Warp kernels
│   │   ├── ir_extractor.py   # Core extraction logic
│   │   ├── test_ir_extractor.py  # 7 kernel validation tests
│   │   ├── fixture_kernels.py    # Diverse test kernels (0fbe)
│   │   └── cases/            # Categorized test cases (d623)
│   │       ├── case_arith.py
│   │       ├── case_atomic.py
│   │       ├── case_branch.py
│   │       ├── case_loop.py
│   │       └── case_vec.py
│   ├── synthesis/            # Kernel generation & synthesis
│   │   ├── generator.py      # 10 kernel type generators (12c4 + 9177)
│   │   ├── pipeline.py       # Single-kernel synthesis
│   │   ├── batch_generator.py    # Optimized batch generation
│   │   ├── analyze_dataset.py    # Dataset analytics (82cf)
│   │   └── validate_dataset.py   # Quality validation (3576)
│   └── examples/             # Example kernels and tests
│       ├── poisson_solver.py     # FEM Poisson equation solver
│       ├── test_poisson.py       # Validation tests
│       └── ...
├── data/
│   ├── samples/              # 120 manual + synthesized samples
│   └── selected_samples/     # 50 curated samples from large dataset
├── notes/                    # Technical documentation
│   ├── ir_format.md          # IR structure documentation
│   ├── warp_basics.md        # Warp compilation flow
│   └── data_stats.md         # Dataset statistics
├── tasks/                    # Milestone task files (M1-M5)
├── README.md                 # This file
└── QUICKSTART.md             # Quick start guide (aa30)
```

## Kernel Types

The generator supports 10 distinct kernel patterns:

1. **arithmetic**: Chains of arithmetic operations (+, -, *, /, min, max)
2. **vector**: Vector operations (dot, cross, length, normalize) for vec2/vec3/vec4
3. **matrix**: Matrix operations (mul, transpose, determinant) for mat22/mat33/mat44
4. **control_flow**: Conditional branches (if/else)
5. **math**: Math functions (sin, cos, exp, log, sqrt, abs)
6. **atomic**: Atomic operations (add, sub, min, max, cas)
7. **nested_loop**: Nested for loops (2-4 levels)
8. **multi_conditional**: Multiple conditional branches (if/elif/else)
9. **scalar_param**: Kernels with scalar parameters
10. **loop**: Simple for loops with reductions

## Utilities

### Analyze Dataset
```bash
python3 code/synthesis/analyze_dataset.py /path/to/data
```
Generates statistics: category distribution, size metrics, kernel counts.

### Validate Dataset
```bash
python3 code/synthesis/validate_dataset.py /path/to/data
```
Checks dataset quality: completeness, consistency, file integrity.

### Test Fixtures
```python
from code.extraction.fixture_kernels import add_constant, conditional_scale, struct_math
# Use pre-built test kernels for validation
```

## Data Format

Each generated pair is a JSON file containing:

```json
{
  "python_source": "@wp.kernel\ndef kernel_xyz(...):\n    ...",
  "ir_code": "void kernel_xyz_forward(...) { ... }",
  "kernel_name": "kernel_xyz",
  "category": "arithmetic",
  "metadata": { ... }
}
```

## Performance

- **Generation rate**: ~180-380 pairs/second (single-threaded)
- **Dataset scale**: 10,500+ pairs generated across all branches
- **Batching**: 10-20 kernels per module compile for efficiency

## Milestone Progress

All 5 milestones completed:

- ✅ **M1**: Environment Setup & Warp Basics
- ✅ **M2**: IR Extraction Mechanism
- ✅ **M3**: FEM Deep Dive (Poisson Solver)
- ✅ **M4**: Synthesis Pipeline
- ✅ **M5**: Scale Up (Batch Generation)

## Contributing Branches

This merged codebase integrates work from:

- **12c4**: Primary base (10.5k pairs, 7 kernel types, complete pipeline)
- **9177**: 3 additional kernel types (nested_loop, multi_conditional, scalar_param)
- **0fbe**: fixture_kernels.py test suite
- **d623**: Categorized test cases structure
- **82cf**: analyze_dataset.py utility
- **3576**: validate_dataset.py utility
- **aa30**: QUICKSTART.md documentation

## Testing

```bash
# Test IR extraction
python3 code/extraction/test_ir_extractor.py

# Test Poisson solver
python3 code/examples/test_poisson.py

# Test all fixture kernels
python3 -c "from code.extraction.fixture_kernels import *"
```

## Requirements

- Python 3.10+
- warp-lang (NVIDIA Warp)

## License

Follow NVIDIA Warp's license for the underlying JIT compilation technology.

## Merge Notes

See `merge_notes/` directory for detailed analysis of all 16 contributing branches.
