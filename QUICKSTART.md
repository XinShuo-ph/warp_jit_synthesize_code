# Quick Start Guide

## Installation

```bash
pip install warp-lang
```

## Generate Training Data

### Quick Test (5 samples)
```bash
python3 code/synthesis/pipeline.py -n 5 -o data/test
```

### Production Dataset (100 samples)
```bash
python3 code/synthesis/pipeline.py -n 100 -o data/production --seed 42
```

### Large Scale (1000+ samples with batch generator)
```bash
python3 code/synthesis/batch_generator.py --count 1000 --output data/large_batch
```

## Verify Results

### Check generated files
```bash
ls data/test/
cat data/test/synth_0000.json | python3 -m json.tool | head -50
```

### Analyze dataset statistics
```bash
python3 << 'EOF'
import json
from pathlib import Path
from collections import Counter

samples_dir = Path('data/test')
categories = []
for f in samples_dir.glob('*.json'):
    with open(f) as fp:
        data = json.load(fp)
        categories.append(data['metadata']['category'])

for cat, count in sorted(Counter(categories).items()):
    print(f"{cat:20s}: {count}")
EOF
```

## Generate Specific Kernel Types

### Only arithmetic kernels
```bash
python3 code/synthesis/pipeline.py -n 10 -o data/arith -c arithmetic
```

### Only control flow kernels
```bash
python3 code/synthesis/pipeline.py -n 10 -o data/ctrl -c control_flow
```

### Mix of specific types
```bash
python3 code/synthesis/pipeline.py -n 20 -o data/mix -c arithmetic vector matrix
```

## Available Kernel Categories

1. `arithmetic` - Basic arithmetic operations
2. `vector` - Vector operations (dot, cross, length, normalize)
3. `matrix` - Matrix operations (multiply, transpose)
4. `control_flow` - Conditionals (if/else, clamp, step)
5. `math` - Math functions (sin, cos, exp, sqrt)
6. `atomic` - Atomic operations (add, min, max)
7. `nested_loop` - Nested loop patterns
8. `multi_conditional` - Multiple conditional branches
9. `combined` - Loop+conditional+math combined
10. `scalar_param` - Kernels with scalar parameters

## Test IR Extraction

```bash
# Test basic extraction
python3 code/extraction/ir_extractor.py
```

## Run Examples

```bash
# Basic kernel examples
python3 code/examples/test_basic_kernels.py

# Poisson solver (FEM example)
python3 code/examples/poisson_solver.py
python3 code/examples/test_poisson.py
```

## Validate Quality

```bash
# Validate extraction quality
python3 code/extraction/validate_extraction.py

# Validate dataset (requires samples matching pattern)
python3 code/synthesis/validate_dataset.py data/samples
```

## Common Issues

### "ModuleNotFoundError: No module named 'warp'"
```bash
pip install warp-lang
```

### CUDA warnings
Normal on CPU-only systems. Warp will use CPU backend automatically.

### "Permission denied" errors
Check file permissions:
```bash
chmod +x code/synthesis/pipeline.py
chmod +x code/synthesis/batch_generator.py
```

## Performance Tips

- Use `--seed` for reproducible results
- Batch generator includes checkpointing for long runs
- Single-threaded: ~180-380 samples/second
- For 10k+ samples, use batch_generator.py

## Data Format

Each JSON file contains:
- `python_source`: Original Python kernel code
- `cpp_forward`: Generated C++ IR code
- `metadata`: Category, description, parameters, etc.

Example:
```json
{
  "python_source": "@wp.kernel\ndef add(a, b, c): ...",
  "cpp_forward": "void add_cpu_kernel_forward(...) { ... }",
  "metadata": {
    "kernel_name": "add_xyz",
    "category": "arithmetic",
    "description": "Arithmetic kernel with 2 operations",
    "device": "cpu",
    ...
  }
}
```

## Next Steps

1. Generate a test dataset (5-10 samples)
2. Inspect the JSON output
3. Generate a larger dataset (100-1000 samples)
4. Use for LLM training

See `README.md` for full documentation.
