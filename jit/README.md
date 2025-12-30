# JAX JIT Code Synthesis

Automated generation of Python→IR training pairs using Google JAX.

## Quick Start

### Dataset
Pre-generated dataset ready to use:
- **11,538 training pairs** in `data/m5_dataset_final.json`
- Each pair includes: Python code, Jaxpr IR, StableHLO IR, metadata
- 6 categories: arithmetic, array ops, math, reductions, linear algebra, composite

### Generate More Data
```bash
cd code/synthesis
python3 generate_m5_dataset.py  # Generates 10k+ pairs
```

### Extract IR from Your Function
```python
import sys; sys.path.append('code/extraction')
from ir_extractor import IRExtractor
import jax.numpy as jnp

extractor = IRExtractor(ir_type="both")
ir = extractor.extract(lambda x: x ** 2, jnp.array([1.0, 2.0]))

print("Jaxpr:", ir['jaxpr'])
print("StableHLO:", ir['stablehlo'])
```

### Run Tests
```bash
cd code/examples
python3 test_poisson.py  # 10 validation tests

cd ../extraction
python3 test_ir_extractor.py  # 23 IR extraction tests
```

## Project Structure

```
jit/
├── code/
│   ├── examples/       # 5 JAX examples + Poisson solver
│   ├── extraction/     # IR extractor + tests
│   └── synthesis/      # Generator, pipeline, batch generator
├── data/               # 11,538 training pairs (11 MB)
├── notes/              # Documentation (jax_basics, ir_format, data_stats)
└── tasks/              # Task breakdowns (M1-M5)
```

## Documentation

- `instructions_jax.md` - Full project instructions
- `PROJECT_SUMMARY.md` - Complete project summary
- `STATE.md` - Current project state
- `notes/` - Technical documentation

## Requirements

```bash
pip install jax jaxlib numpy
```

## Status

✅ All 5 milestones complete  
✅ 11,538 training pairs generated  
✅ All tests passing  
✅ Production ready
