# Warp JIT Code Synthesis - Branch `cursor/instructions-wrapup-completion-b74a`

## Progress Summary
- **Milestone reached**: M5 (Complete)
- **Key deliverables**:
  - IR extraction pipeline for Warp kernels
  - Python→IR code pair generator with 7 kernel types
  - Batch generation with resume capability
  - 371+ validated sample pairs
  - FEM-based Poisson solver example

## What Works
- **IR Extraction** (`code/extraction/ir_extractor.py`): Extracts C++ code from any Warp kernel, supporting all data types (float, vec3, mat33, etc.)
- **Synthesis Pipeline** (`code/synthesis/pipeline.py`): Generates valid Python→IR pairs across 7 kernel types
- **Batch Generator** (`code/synthesis/batch_generator.py`): Scales to 10k+ pairs with resumable state
- **Poisson Solver** (`code/examples/poisson_solver.py`): FEM-based PDE solver demonstrating `warp.fem` capabilities
- **Test Suite**: All tests pass (5/5 IR tests, 4/4 Poisson tests)

## Requirements
```bash
pip install warp-lang
```

## Quick Start
```bash
# Extract IR from a kernel
python3 code/extraction/ir_extractor.py

# Run IR extraction tests
python3 code/extraction/test_ir_extractor.py

# Generate 10 Python→IR pairs
python3 code/synthesis/pipeline.py --count 10

# Run Poisson solver tests
python3 code/examples/test_poisson.py
```

## File Structure
```
jit/
├── code/
│   ├── extraction/          # IR extraction from Warp kernels
│   │   ├── ir_extractor.py      # Core extraction: extract_ir(), get_kernel_source()
│   │   └── test_ir_extractor.py # 5 test cases (arithmetic, loop, conditional, matrix, math)
│   ├── synthesis/           # Python→IR pair generation
│   │   ├── generator.py         # KernelGenerator: 7 kernel types
│   │   ├── pipeline.py          # End-to-end synthesis pipeline
│   │   └── batch_generator.py   # Resumable batch generation
│   └── examples/            # Working Warp examples
│       ├── ex1_basic_kernel.py  # Vector addition
│       ├── ex2_math_ops.py      # Math functions
│       ├── ex3_vec_types.py     # Vector types
│       ├── poisson_solver.py    # FEM-based Poisson equation solver
│       └── test_poisson.py      # 4 validation tests
├── data/
│   ├── generated/           # Batch-generated pairs (266 samples)
│   └── samples/             # Pipeline-generated pairs (105 samples)
├── notes/
│   ├── warp_basics.md       # Warp compilation flow documentation
│   ├── ir_format.md         # IR structure and patterns
│   └── data_stats.md        # Dataset statistics
└── tasks/                   # Milestone task definitions
```

## Generated Data Format
```json
{
  "kernel_name": "arith_3pm0gb",
  "kernel_type": "arithmetic",
  "python_source": "@wp.kernel\ndef arith_3pm0gb(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):\n    i = wp.tid()\n    c[i] = ((a[i] / b[i]) / b[i])\n",
  "ir_code": "#define WP_TILE_BLOCK_DIM 1\n#define WP_NO_CRT\n#include \"builtin.h\"\n...",
  "device": "cpu"
}
```

## Kernel Types Supported
| Type | Description | Example Operation |
|------|-------------|-------------------|
| `arithmetic` | Basic math ops | `c[i] = a[i] + b[i] * 2.0` |
| `math` | Math functions | `y[i] = wp.sin(x[i]) + wp.exp(-x[i])` |
| `loop` | Explicit loops | `for j in range(n): total += a[j]` |
| `conditional` | If/else branches | `if x[i] > 0: y[i] = x[i]` |
| `vector` | vec3 operations | `c[i] = a[i] + wp.normalize(b[i])` |
| `matrix` | mat33 operations | `out[i] = A[i] @ v[i]` |
| `combined` | Mixed patterns | Multiple patterns combined |

## IR Pattern Reference
| Python | C++ IR |
|--------|--------|
| `wp.tid()` | `builtin_tid1d()` |
| `a[i]` | `wp::load(wp::address(var_a, var_i))` |
| `a[i] = x` | `wp::array_store(var_a, var_i, var_x)` |
| `a + b` | `wp::add(var_a, var_b)` |
| `for i in range(n)` | `wp::range_t`, loop construct |
| `if cond` | C++ if with `bool var_N` |
| `wp.sin(x)` | `wp::sin(var_x)` |

## Known Issues / TODOs
- **No CUDA testing**: Environment has no GPU; all tests run on CPU only
- **Import path**: Fixed hardcoded path to warp fem utils (now uses dynamic path)
- **Scaling**: Batch generation is slow (~0.3 pairs/s) due to JIT compilation overhead

## Scaling to 10k+ Pairs
```bash
cd jit/code/synthesis
python3 batch_generator.py --count 10000 --output ../../data/generated --resume
```
Estimated time: ~9 hours at 0.3 pairs/second (CPU-only mode)
