# Quick Reference: CUDA Backend Usage

## For GPU Users

### 1. Install Dependencies
```bash
pip install warp-lang
```

### 2. Generate CUDA Kernels
```bash
cd cuda/code/synthesis

# Small test (10 kernels)
python3 pipeline.py -n 10 -d cuda -o /output/dir

# Production (10,000 kernels)
python3 batch_generator.py -n 10000 -d cuda -o /output/dir
```

### 3. Run Tests
```bash
cd cuda/tests
bash run_all_cuda_tests.sh
```

## Command Line Options

### pipeline.py
```bash
python3 pipeline.py \
  -n 100              # Number of kernels to generate
  -d cuda             # Device: cpu or cuda
  -o /output/dir      # Output directory
  -s 42               # Random seed (optional)
  -c math vector      # Categories to generate (optional)
```

### batch_generator.py
```bash
python3 batch_generator.py \
  -n 10000            # Number of kernels
  -d cuda             # Device: cpu or cuda
  -o /output/dir      # Output directory
  -s 42               # Random seed (optional)
```

## Python API

### Extract IR with Device Parameter
```python
from ir_extractor import extract_ir

# CUDA IR
result = extract_ir(kernel, device="cuda", include_backward=True)

# Access results
python_src = result["python_source"]
cuda_fwd = result["forward_code"]
cuda_bwd = result["backward_code"]  # if include_backward=True
metadata = result["metadata"]
```

### Generate Kernels
```python
from generator import generate_kernel, GENERATORS

# Generate single kernel
spec = generate_kernel("arithmetic", seed=42)

# Available categories
print(GENERATORS.keys())
# dict_keys(['arithmetic', 'vector', 'matrix', 'control_flow', 'math', 'atomic'])
```

### Pipeline Usage
```python
from pipeline import synthesize_batch

# Generate batch
pairs = synthesize_batch(
    n=100,
    categories=["math", "vector"],
    seed=42,
    device="cuda"
)

# Process results
for pair in pairs:
    python_code = pair["python_source"]
    cuda_code = pair["cpp_forward"]
    category = pair["metadata"]["category"]
```

## Output Format

Each generated JSON file contains:

```json
{
  "python_source": "Python kernel source code",
  "cpp_forward": "CUDA C forward kernel",
  "metadata": {
    "kernel_name": "kernel_name",
    "category": "arithmetic|vector|matrix|control_flow|math|atomic",
    "description": "Human-readable description",
    "device": "cuda",
    "seed": 42
  }
}
```

## Kernel Categories

| Category | Operations | Example |
|----------|-----------|---------|
| `arithmetic` | +, -, *, /, min, max | `c[i] = a[i] + b[i]` |
| `math` | sin, cos, exp, log, sqrt | `out[i] = sin(cos(x[i]))` |
| `vector` | dot, cross, normalize | `out[i] = dot(a[i], b[i])` |
| `matrix` | mat*vec, mat*mat, transpose | `out[i] = m[i] * v[i]` |
| `control_flow` | if/else, for loops | `if (x[i] > 0) out[i] = x[i]` |
| `atomic` | atomic_add, atomic_min/max | `atomic_add(result, 0, x[i])` |

## CUDA Patterns to Expect

### Thread Indexing
```cpp
for (size_t _idx = blockDim.x * blockIdx.x + threadIdx.x;
     _idx < dim.size;
     _idx += blockDim.x * gridDim.x)
{
    // Kernel body
}
```

### Shared Memory
```cpp
wp::tile_shared_storage_t tile_mem;
wp::tile_shared_storage_t::init();
```

### Atomic Operations
```cpp
wp::atomic_add(result, 0, value);
wp::atomic_max(result, 0, value);
```

## Testing

### Quick Test (No GPU Required)
```bash
cd cuda/code/examples
python3 test_cuda_codegen.py
```

### Full Test Suite (GPU Recommended)
```bash
cd cuda/tests
bash run_all_cuda_tests.sh
```

### Manual Verification
```bash
# Generate 5 kernels
python3 pipeline.py -n 5 -d cuda -o /tmp/test_output

# Check for CUDA patterns
grep -r "blockIdx\|threadIdx" /tmp/test_output/
```

## Troubleshooting

### "CUDA driver not found"
- Expected on CPU-only machines
- Code generation still works
- Copy to GPU machine for execution

### Empty output directory
- Check disk space
- Check write permissions
- Verify warp-lang installed

### Import errors
```python
# Add paths if needed
import sys
from pathlib import Path
sys.path.insert(0, str(Path("cuda/code/extraction")))
sys.path.insert(0, str(Path("cuda/code/synthesis")))
```

## Performance Tips

### Code Generation
- Use batch_generator.py for >100 kernels
- Generation speed: ~300-350 pairs/sec (CPU)
- Bottleneck: Kernel compilation, not execution

### GPU Execution
- Compilation time similar on CPU/GPU
- Execution time much faster on GPU
- Overall throughput limited by compilation

## Examples

### Generate Math Kernels Only
```bash
python3 pipeline.py -n 100 -d cuda -c math -o /data/math_kernels
```

### Generate with Specific Seed
```bash
python3 pipeline.py -n 50 -d cuda -s 12345 -o /data/reproducible
```

### Large Scale Generation
```bash
python3 batch_generator.py -n 100000 -d cuda -o /data/large_dataset
```

## Integration Example

```python
import json
from pathlib import Path

# Load generated data
dataset = []
for file in Path("/data/cuda_kernels").glob("*.json"):
    with open(file) as f:
        pair = json.load(f)
        dataset.append({
            "input": pair["python_source"],
            "target": pair["cpp_forward"],
            "category": pair["metadata"]["category"]
        })

# Use in training
for sample in dataset:
    train_model(
        input_code=sample["input"],
        target_ir=sample["target"]
    )
```

## Documentation Links

- Full documentation: `cuda/README_CUDA.md`
- Project summary: `cuda/PROJECT_SUMMARY.md`
- State tracking: `cuda/CUDA_STATE.md`
- Test suite: `cuda/tests/`
- Examples: `cuda/code/examples/`
