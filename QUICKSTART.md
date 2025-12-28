# Quick Start Guide

## Running the Examples

### Basic Warp Examples
```bash
cd /workspace
python3 code/examples/01_simple_kernel.py
python3 code/examples/02_vector_ops.py
python3 code/examples/03_control_flow.py
```

### Poisson Solver
```bash
# Run solver
python3 code/examples/poisson_solver.py

# Run validation tests
python3 code/examples/test_poisson.py
```

## IR Extraction

### Extract IR from existing test cases
```python
from code.extraction.ir_extractor import IRExtractor

extractor = IRExtractor()
# After compiling a kernel:
ir_data = extractor.extract_ir(my_kernel)
print(ir_data['forward_function'])
```

## Generate Training Data

### Generate a batch of samples
```python
from code.synthesis.pipeline import SynthesisPipeline

pipeline = SynthesisPipeline(output_dir="data/my_samples")
pairs = pipeline.generate_batch(count=50, verbose=True)
```

### Generate specific kernel type
```python
from code.synthesis.generator import KernelGenerator, KernelSpec, OpType

generator = KernelGenerator()
spec = KernelSpec(
    name="my_kernel",
    op_type=OpType.ARITHMETIC,
    num_inputs=2,
    num_outputs=1,
    has_scalar_param=True,
    complexity=2
)
code = generator.generate_kernel(spec)
print(code)
```

## Dataset Statistics

```bash
# Count samples by type
python3 << 'EOF'
import json
from pathlib import Path
from collections import Counter

samples_dir = Path('data/samples')
types = []
for f in samples_dir.glob('*.json'):
    with open(f) as fp:
        data = json.load(fp)
        types.append(data['metadata']['op_type'])

for op_type, count in sorted(Counter(types).items()):
    print(f"{op_type:15s}: {count}")
EOF
```

## Key Files

- `code/extraction/ir_extractor.py` - Main IR extraction utility
- `code/synthesis/generator.py` - Kernel code generator
- `code/synthesis/pipeline.py` - End-to-end synthesis pipeline
- `notes/warp_basics.md` - Kernel compilation documentation
- `notes/ir_format.md` - IR structure documentation

## Extending the Generator

To add new kernel types:
1. Add enum to `OpType` in `generator.py`
2. Implement `_generate_[type]()` method
3. Update `_create_dummy_inputs()` in `pipeline.py`
4. Test with a small batch

## Performance Notes

- Each kernel compilation takes ~1.2 seconds
- 120 samples generated in ~2.5 minutes
- For large-scale generation (10k+), consider:
  - Parallel generation with multiprocessing
  - Batch compilation
  - Caching compiled modules
