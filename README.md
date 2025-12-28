# NVIDIA Warp Python→IR Synthesis Pipeline

A complete pipeline for generating Python Warp kernel code and extracting corresponding C++ IR representations for training machine learning models.

## Quick Start

```bash
# Install dependencies
pip install warp-lang

# Generate training pairs
python3 code/synthesis/pipeline.py -n 100 -o data/samples

# View generated data
ls data/samples/
```

## Features

- **10 Kernel Types**: arithmetic, vector, matrix, control_flow, math, atomic, nested_loop, multi_conditional, combined, scalar_param
- **IR Extraction**: Forward and backward pass C++ code extraction
- **Batch Generation**: Optimized large-scale data generation (~180 pairs/sec)
- **Validation**: Comprehensive test suites for verification

## Project Structure

```
code/
├── extraction/          # IR extraction utilities
│   └── ir_extractor.py  # Core IR extraction
├── synthesis/           # Kernel generation
│   ├── generator.py     # 10 kernel type generators
│   ├── pipeline.py      # End-to-end synthesis
│   └── batch_generator.py  # Large-scale generation
└── examples/            # Example scripts
tests/
├── cases/               # Categorized test kernels
└── fixture_kernels.py   # Test fixtures with structs
notes/                   # Documentation
data/                    # Generated data (gitignored)
```

## Data Format

Each generated pair is a JSON file:
```json
{
  "python_source": "@wp.kernel\ndef add_kernel(...): ...",
  "cpp_forward": "// C++ forward pass code",
  "metadata": {"category": "arithmetic", ...}
}
```

## API Reference

### Generator
```python
from code.synthesis.generator import generate_kernel, GENERATORS

# Generate random kernel
spec = generate_kernel(category="arithmetic", seed=42)
print(spec.source)
print(spec.category)
```

### Pipeline CLI
```bash
python3 code/synthesis/pipeline.py -n 1000 -o output/ --seed 42
```

## Merge Sources

This codebase merges contributions from 16 development branches:
- **12c4**: Base pipeline (6 types), batch generation, documentation
- **9177**: 4 additional kernel types (nested_loop, multi_conditional, combined, scalar_param)
- **3576**: Categorized test cases
- **0fbe**: Fixture kernels with struct examples
- **aa30**: QUICKSTART guide
