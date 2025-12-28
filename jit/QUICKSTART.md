# Quick Start Guide

## Installation

```bash
pip install warp-lang
```

## Basic Usage

### Generate Training Samples
```bash
cd /workspace/jit

# Generate 10 samples
python3 code/synthesis/pipeline.py -n 10 -o data/samples

# Generate with specific seed for reproducibility
python3 code/synthesis/pipeline.py -n 100 -o data/samples -s 42
```

### Test IR Extraction
```bash
python3 code/extraction/ir_extractor.py
```

### Run Poisson Solver Example
```bash
python3 code/examples/poisson_solver.py
python3 code/examples/test_poisson.py
```

## IR Extraction API

```python
import warp as wp
from code.extraction.ir_extractor import extract_ir

@wp.kernel
def my_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = a[tid] * 2.0

# Trigger compilation
wp.init()

# Extract IR
result = extract_ir(my_kernel)
print(result['python_source'])
print(result['forward_code'])
```

## Kernel Generation API

```python
from code.synthesis.generator import generate_kernel, GENERATORS

# See available categories
print(list(GENERATORS.keys()))
# ['arithmetic', 'vector', 'matrix', 'control_flow', 'math', 'atomic',
#  'nested_loop', 'multi_condition', 'combined', 'scalar_param']

# Generate specific category
spec = generate_kernel(category="vector", seed=42)
print(spec.source)

# Generate random category
spec = generate_kernel(seed=123)
print(f"Category: {spec.category}")
print(spec.source)
```

## Batch Generation

For large-scale dataset generation:

```bash
# Generate 1000 samples with progress and checkpointing
python3 code/synthesis/batch_generator.py --count 1000 --output data/large
```

## Dataset Statistics

```python
import json
from pathlib import Path
from collections import Counter

samples_dir = Path('data/samples')
categories = []
for f in samples_dir.glob('*.json'):
    with open(f) as fp:
        data = json.load(fp)
        categories.append(data['metadata']['category'])

for cat, count in sorted(Counter(categories).items()):
    print(f"{cat:20s}: {count}")
```

## Key Files

| File | Description |
|------|-------------|
| `code/extraction/ir_extractor.py` | IR extraction from Warp kernels |
| `code/synthesis/generator.py` | 10-type kernel generator |
| `code/synthesis/pipeline.py` | End-to-end synthesis pipeline |
| `code/synthesis/batch_generator.py` | Large-scale batch generation |
| `code/examples/poisson_solver.py` | FEM Poisson solver example |
| `notes/ir_format.md` | IR structure documentation |

## Adding New Kernel Types

To add a new kernel category:

1. Create a generator function in `generator.py`:
```python
def generate_my_type_kernel(seed: int | None = None) -> KernelSpec:
    if seed is not None:
        random.seed(seed)
    
    name = random_name("mytype")
    source = f'''@wp.kernel
def {name}(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a[tid] * 2.0
'''
    return KernelSpec(
        name=name,
        category="my_type",
        source=source,
        arg_types={"a": "wp.array(dtype=float)", "out": "wp.array(dtype=float)"},
        description="My custom kernel type",
        metadata={"seed": seed}
    )
```

2. Add to GENERATORS dict:
```python
GENERATORS = {
    # ... existing ...
    "my_type": generate_my_type_kernel,
}
```

3. Test:
```bash
python3 code/synthesis/pipeline.py -n 5 -c my_type
```

## Performance Notes

- Each kernel compilation: ~1.2 seconds (CPU mode)
- 100 samples: ~2 minutes
- Generation rate: ~180 pairs/second
- For 10k+ samples, use batch_generator with checkpointing
