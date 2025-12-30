# JAX Code Synthesis - Quick Reference

## Installation
```bash
pip install jax jaxlib matplotlib
```

## Generate Dataset

### Small Test (100 samples)
```bash
cd jit
python code/synthesis/pipeline.py --count 100 --output data/test
```

### Medium Dataset (1000 samples)
```bash
python code/synthesis/batch_generator.py \
    --target 1000 \
    --output data/dataset_1k \
    --validate
```

### Large Dataset (10k samples)
```bash
python code/synthesis/batch_generator.py \
    --target 10000 \
    --output data/dataset_10k \
    --checkpoint-every 2000 \
    --validate
```

## Analyze Dataset
```bash
python code/synthesis/analyze_dataset.py data/dataset_10k --save stats.md
```

## Custom IR Extraction

### Python API
```python
from code.extraction.ir_extractor import IRExtractor
import jax.numpy as jnp

extractor = IRExtractor(dialect='stablehlo')

def my_func(x, y):
    return jnp.dot(x, y) + jnp.tanh(x)

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

result = extractor.extract_with_metadata(my_func, x, y)
print(result['ir_code'])
```

### Command Line (using examples)
```bash
python code/extraction/ir_extractor.py
python code/extraction/test_ir_extractor.py
```

## Run Examples

### Basic JAX Features
```bash
python code/examples/01_basic_jit.py
python code/examples/02_autodiff.py
python code/examples/03_vmap.py
python code/examples/04_ir_extraction.py
```

### Scientific Computing
```bash
python code/examples/poisson_solver.py
python code/examples/heat_equation.py
python code/examples/gradient_descent.py
```

## Dataset Format

Each JSON file contains:
```json
{
  "function_name": "arith_add_0",
  "python_source": "def arith_add_0(x, y):\n    return x + y\n",
  "ir_code": "module @jit_arith_add_0 ...",
  "dialect": "stablehlo",
  "category": "arithmetic",
  "operation": "add",
  "input_info": [...]
}
```

## Categories & Operations

**Arithmetic** (4): add, sub, mul, div  
**Math** (5): sin, cos, exp, tanh, sqrt  
**Array** (5): dot, matmul, sum, mean, transpose  
**Control Flow** (3): where, maximum, minimum  
**Combined** (6): linear, quadratic, sigmoid, softmax, relu, mse  

Total: **20 unique operations**

## Performance

- Generation Rate: **239 samples/sec**
- Success Rate: **100%**
- 10k Dataset: **42 seconds**
- Memory: Minimal (~500MB for 10k samples)

## Directory Structure

```
jit/
├── README.md                 # Full documentation
├── QUICKSTART.md            # This file
├── JAX_STATE.md             # Project status
├── code/
│   ├── examples/            # JAX examples (7 files)
│   ├── extraction/          # IR extraction (3 files)
│   └── synthesis/           # Pipeline (4 files)
├── data/
│   ├── dataset_10k/         # 10k production dataset
│   └── samples_m4/          # 100 validation samples
├── notes/
│   ├── jax_basics.md        # JAX guide
│   ├── ir_format.md         # StableHLO format
│   └── data_stats.md        # Dataset stats
└── tasks/                   # Task files (M1-M5)
```

## Key Files

- `code/extraction/ir_extractor.py` - Core IR extraction
- `code/synthesis/generator.py` - Kernel generation
- `code/synthesis/pipeline.py` - Single/batch generation
- `code/synthesis/batch_generator.py` - Large-scale generation
- `code/synthesis/analyze_dataset.py` - Dataset analysis

## Common Issues

**Q: No GPU available?**  
A: JAX will use CPU backend automatically. For GPU, install `jax[cuda]`.

**Q: Generation too slow?**  
A: Use `--checkpoint-every` to save progress. Current rate: 239 samples/sec.

**Q: Customize categories?**  
A: Edit `generator.py` to add new operation types.

## Testing

Run all tests:
```bash
python code/extraction/test_ir_extractor.py
python code/examples/poisson_solver.py
python code/synthesis/generator.py
```

Expected: All tests pass ✓

## Citation

If using JAX IR data for research:
```
JAX: https://github.com/google/jax
StableHLO: https://github.com/openxla/stablehlo
```
