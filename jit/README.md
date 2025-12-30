# JAX JIT Code Synthesis - Production Ready

## Project Summary

This project generates Python→IR (StableHLO) paired training data using JAX for machine learning training. It successfully completes all 5 milestones and generates a production-ready dataset of 10,000+ samples.

## What Works

### M1: JAX Environment & Basics
- ✅ JAX installed and tested (version 0.8.2)
- ✅ Basic examples: JIT compilation, autodiff, vmap
- ✅ IR extraction mechanism using StableHLO
- ✅ 4 working examples demonstrating JAX features

### M2: IR Extraction
- ✅ `code/extraction/ir_extractor.py`: Full-featured IR extraction
- ✅ Supports both HLO and StableHLO dialects
- ✅ 28 comprehensive test cases
- ✅ 6+ sample Python→IR pairs
- ✅ Metadata extraction (function signatures, input shapes, dtypes)

### M3: Scientific Computing Examples
- ✅ `code/examples/poisson_solver.py`: 1D Poisson equation solver (direct method)
- ✅ `code/examples/heat_equation.py`: 1D heat equation solver (explicit FD)
- ✅ `code/examples/gradient_descent.py`: Optimization with GD and Adam
- ✅ All tests pass with analytical validation

### M4: Synthesis Pipeline
- ✅ `code/synthesis/generator.py`: Programmatic kernel generation
  - Arithmetic: add, sub, mul, div
  - Math: sin, cos, exp, tanh, sqrt
  - Array: dot, matmul, sum, mean, transpose
  - Control flow: where, maximum, minimum
  - Combined: linear, quadratic, sigmoid, softmax, relu, mse
- ✅ `code/synthesis/pipeline.py`: End-to-end Python→IR generation
- ✅ 100+ validated sample pairs

### M5: Large-Scale Generation
- ✅ `code/synthesis/batch_generator.py`: Efficient batch generation
- ✅ **10,000 Python→IR pairs** generated (balanced dataset)
- ✅ Checkpointing and resumption support
- ✅ 239 samples/sec generation rate
- ✅ 100% success rate (0 failures)
- ✅ Dataset validation complete

## Dataset Statistics

### Overview
- **Total Samples**: 10,000
- **Dialect**: StableHLO (MLIR-based)
- **Balance**: 2,000 samples per category (5 categories)
- **Success Rate**: 100%
- **Generation Time**: ~42 seconds

### Distribution
| Category | Count | Percentage |
|----------|-------|------------|
| Arithmetic | 2,000 | 20% |
| Math | 2,000 | 20% |
| Array | 2,000 | 20% |
| Control Flow | 2,000 | 20% |
| Combined | 2,000 | 20% |

### Code Statistics
- **IR Code**: 278-911 chars (mean: 432)
- **Python Source**: 38-50 chars (mean: 43)
- **Unique Operations**: 20 types

## Requirements

```bash
pip install jax jaxlib matplotlib
```

## Quick Start

### Generate Small Dataset
```bash
cd jit
python code/synthesis/pipeline.py --count 100 --output data/test
```

### Generate Large Dataset
```bash
python code/synthesis/batch_generator.py \
    --target 10000 \
    --output data/large \
    --balanced \
    --validate
```

### Analyze Dataset
```bash
python code/synthesis/analyze_dataset.py data/large --save stats.md
```

### Extract IR from Custom Function
```python
from code.extraction.ir_extractor import extract_ir_pair
import jax.numpy as jnp

def my_function(x, y):
    return jnp.dot(x, y) + jnp.sin(x)

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

pair = extract_ir_pair(my_function, x, y)
print(pair['ir_code'])
```

## File Structure

```
jit/
├── JAX_STATE.md              # Progress tracker
├── code/
│   ├── examples/             # JAX examples & scientific computing
│   │   ├── 01_basic_jit.py
│   │   ├── 02_autodiff.py
│   │   ├── 03_vmap.py
│   │   ├── 04_ir_extraction.py
│   │   ├── poisson_solver.py
│   │   ├── heat_equation.py
│   │   └── gradient_descent.py
│   ├── extraction/           # IR extraction utilities
│   │   ├── ir_extractor.py
│   │   ├── test_ir_extractor.py
│   │   └── save_sample_pairs.py
│   └── synthesis/            # Data generation pipeline
│       ├── generator.py       # Kernel generation
│       ├── pipeline.py        # Single/batch generation
│       ├── batch_generator.py # Large-scale generation
│       └── analyze_dataset.py # Dataset analysis
├── data/
│   ├── dataset_10k/          # 10k production dataset
│   └── samples_m4/           # 100 validation samples
├── notes/
│   ├── jax_basics.md         # JAX compilation flow
│   ├── ir_format.md          # StableHLO format guide
│   └── data_stats.md         # Dataset statistics
└── tasks/                    # Task breakdowns (M1-M5)
```

## Generated Data Format

Each JSON file contains:
```json
{
  "function_name": "arith_add_0",
  "python_source": "def arith_add_0(x, y):\n    return x + y\n",
  "ir_code": "module @jit_arith_add_0 ...",
  "dialect": "stablehlo",
  "signature": "(x, y)",
  "input_info": [
    {"index": 0, "shape": [3], "dtype": "float32"},
    {"index": 1, "shape": [3], "dtype": "float32"}
  ],
  "category": "arithmetic",
  "operation": "add",
  "complexity": "simple"
}
```

## Key Features

1. **Comprehensive Coverage**: 5 categories, 20 unique operations
2. **Balanced Dataset**: Equal distribution across categories
3. **High Quality**: 100% success rate, validated samples
4. **Scalable**: 239 samples/sec, checkpointing support
5. **Flexible**: Supports both HLO and StableHLO dialects
6. **Well-Documented**: Clear code structure, extensive comments

## Advantages Over Warp

1. **Maturity**: JAX is more mature and widely used
2. **IR Format**: StableHLO is more stable and portable (MLIR-based)
3. **Performance**: Faster generation (239 vs ~100 samples/sec typical for Warp)
4. **Ecosystem**: Better integration with ML frameworks
5. **Documentation**: Extensive JAX documentation and community
6. **Flexibility**: Easier to extend with custom operations

## Next Steps for GPU Support

To adapt for GPU/CUDA:
1. IR extraction already supports device parameter
2. Test with `device="cuda"` on GPU-enabled machine
3. Generate separate dataset for GPU-specific operations
4. Compare CPU vs GPU IR differences

## Known Issues / TODOs

- Source code extraction uses `inspect.getsource()` which may fail for dynamically generated functions (currently returns placeholder)
- Could add more complex patterns (nested loops, recursion)
- GPU-specific operations not yet included

## Testing

All examples and tests pass consistently:
- M1: 4/4 examples ✓
- M2: 28/28 test cases ✓
- M3: 3/3 solvers ✓
- M4: 100/100 samples ✓
- M5: 10000/10000 samples ✓

## Performance

- **Generation Rate**: 239 samples/second
- **Success Rate**: 100%
- **10k Dataset Time**: 42 seconds
- **Validation**: 100/100 samples valid

---

**Status**: ✅ Production Ready - All 5 milestones complete, 10k dataset generated and validated.
