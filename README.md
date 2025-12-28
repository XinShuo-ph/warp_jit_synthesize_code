# Warp JIT Code Synthesis - cursor/instructions-wrapup-completion-58eb

## Progress Summary
- **Milestone reached**: M4 Complete (Ready for M5)
- **Key deliverables**:
  - IR extraction mechanism for Warp kernels
  - Python→C++ synthesis pipeline
  - 104 training data samples (Python→IR pairs)
  - Poisson equation FEM solver with tests
  - Comprehensive documentation of IR format

## What Works
- **IR Extraction** (`code/extraction/ir_extractor.py`): Extracts C++ intermediate representation from compiled Warp kernels. Supports CPU device, returns structured results with metadata.
- **Synthesis Pipeline** (`code/synthesis/pipeline.py`): End-to-end generation of Python kernel → C++ IR pairs. Generates varied kernel types with 100% success rate.
- **Kernel Generator** (`code/synthesis/generator.py`): Programmatic generation of 7 kernel categories: arithmetic, vector, conditional, loop, function, reduction, and math operations.
- **Poisson Solver** (`code/examples/poisson_solver.py`): 2D Poisson equation solver using Warp's FEM module with manufactured solution validation.
- **Examples**: Basic, vector, functions, and IR extraction examples.

## Requirements

```bash
pip install warp-lang
```

Tested with:
- Python 3.12
- warp-lang 1.10.1

## Quick Start

```bash
# Install dependencies
pip install warp-lang

# Test the synthesis pipeline (generates 10 samples)
python3 code/synthesis/pipeline.py

# Run IR extractor tests
python3 code/extraction/test_extractor.py

# Run Poisson solver tests
python3 code/examples/test_poisson.py

# Run basic example
python3 code/examples/example_01_basic.py
```

## File Structure

```
/workspace/
├── code/
│   ├── extraction/
│   │   ├── ir_extractor.py      # Core IR extraction API
│   │   └── test_extractor.py    # Test suite for extractor
│   ├── synthesis/
│   │   ├── generator.py         # Kernel code generator (7 types)
│   │   ├── pipeline.py          # End-to-end synthesis pipeline
│   │   ├── generate_dataset.py  # Dataset generation script
│   │   └── validate_dataset.py  # Dataset validation
│   └── examples/
│       ├── example_01_basic.py  # Basic kernel example
│       ├── example_02_vectors.py # Vector operations
│       ├── example_03_functions.py # Helper functions
│       ├── example_04_ir_extraction.py # IR extraction demo
│       ├── poisson_solver.py    # 2D Poisson FEM solver
│       └── test_poisson.py      # Poisson solver tests (5 tests)
├── data/
│   └── samples/                 # 104 generated Python→IR pairs
│       ├── sample_00001.py      # Python kernel source
│       ├── sample_00001.cpp     # Generated C++ IR
│       └── sample_00001.json    # Metadata
├── notes/
│   ├── warp_basics.md           # Warp compilation flow docs
│   ├── ir_format.md             # IR structure documentation
│   └── gpu_analysis.md          # GPU/CUDA analysis (if present)
└── tasks/
    ├── m1_tasks.md              # M1: Environment setup tasks
    ├── m2_tasks.md              # M2: IR extraction tasks
    ├── m3_tasks.md              # M3: FEM deep dive tasks
    └── m4_tasks.md              # M4: Synthesis pipeline tasks
```

## Generated Data Format

Each sample consists of three files:

**Python source** (`sample_XXXXX.py`):
```python
import warp as wp

@wp.kernel
def math_0001(data: wp.array(dtype=float),
           scale: float,
           output: wp.array(dtype=float)):
    i = wp.tid()
    val = data[i] * scale
    output[i] = wp.pow(val, 2.0)
```

**C++ IR** (`sample_XXXXX.cpp`):
```cpp
#include "builtin.h"

struct wp_args_math_0001_hash {
    wp::array_t<wp::float32> data;
    wp::float32 scale;
    wp::array_t<wp::float32> output;
};

void math_0001_hash_cpu_kernel_forward(...) {
    // SSA-form C++ code
    wp::int32 var_0 = builtin_tid1d();
    wp::float32 var_1 = wp::address(var_data, var_0);
    // ...
}
```

**Metadata** (`sample_XXXXX.json`):
```json
{
  "sample_id": 1,
  "kernel_name": "math_0001",
  "category": "math",
  "complexity": "simple",
  "module_name": "temp_kernel_1",
  "module_hash": "5dced2d",
  "python_file": "sample_00001.py",
  "cpp_file": "sample_00001.cpp",
  "python_size": 217,
  "cpp_size": 7491
}
```

## Kernel Categories

| Category | Description | Example Operations |
|----------|-------------|-------------------|
| arithmetic | Element-wise math | `a + b * c` |
| vector | Vector operations | `wp.normalize(v1 + v2 * dt)` |
| conditional | Branching logic | `if/elif/else` |
| loop | Iteration patterns | `for j in range(n)` |
| function | Helper functions | `@wp.func` definitions |
| reduction | Aggregation patterns | Sum, accumulate |
| math | Math functions | `wp.sin`, `wp.exp`, `wp.sqrt` |

## Known Issues / TODOs

- **GPU support**: IR extractor has `device` parameter but not tested with CUDA (no GPU in test environment)
- **Cache sensitivity**: IR extraction can fail if stale cache entries exist; clear cache if issues occur
- **M5 not started**: Scale-up to 10k+ samples planned but not implemented

## API Reference

### IR Extraction

```python
from extraction.ir_extractor import extract_ir, extract_ir_to_file

# Extract IR as structured result
result = extract_ir(my_kernel, device="cpu", force_compile=True)
if result.success:
    print(result.cpp_source)  # The C++ IR
    print(result.python_source)  # Original Python

# Save to files
extract_ir_to_file(my_kernel, output_dir="/path/to/output")
```

### Kernel Generation

```python
from synthesis.generator import KernelGenerator

generator = KernelGenerator(seed=42)

# Generate single kernel
spec = generator.generate_math_kernel()
print(spec.source_code)

# Generate batch
specs = generator.generate_batch(100, categories=['math', 'loop'])
```

### Synthesis Pipeline

```python
from synthesis.pipeline import SynthesisPipeline

pipeline = SynthesisPipeline(output_dir="/path/to/output")
stats = pipeline.generate_dataset(count=100)
print(f"Success rate: {stats['success_rate']:.1%}")
```
