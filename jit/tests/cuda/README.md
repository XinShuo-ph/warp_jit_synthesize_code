# CUDA Test Suite

This directory contains tests for the CUDA backend of the Warp JIT code synthesis pipeline.

## Test Files

| File | Description | GPU Required |
|------|-------------|--------------|
| `test_extraction.py` | Tests CUDA IR extraction | No |
| `test_kernels.py` | Tests GPU kernel execution | Yes |
| `run_gpu_tests.sh` | Runs all tests | Yes (for full suite) |

## Running Tests

### On a Machine Without GPU

You can still run the extraction tests to verify CUDA IR generation:

```bash
cd /workspace/jit
pip install warp-lang pytest
python3 -m pytest tests/cuda/test_extraction.py -v
```

These tests verify that:
- CUDA IR can be extracted for all kernel types (forward + backward)
- CPU and CUDA IR are different
- CUDA-specific markers (blockDim, threadIdx) are present

### On a Machine With GPU

Run the full test suite:

```bash
cd /workspace/jit
chmod +x tests/cuda/run_gpu_tests.sh
./tests/cuda/run_gpu_tests.sh
```

Or run tests individually:

```bash
# All CUDA tests
python3 -m pytest tests/cuda/ -v

# Just extraction tests
python3 -m pytest tests/cuda/test_extraction.py -v

# Just GPU execution tests
python3 -m pytest tests/cuda/test_kernels.py -v
```

## Test Categories

### IR Extraction Tests (`test_extraction.py`)

These tests verify CUDA code generation without requiring a GPU:

- **Forward extraction**: Verifies forward kernel IR is generated
- **Backward extraction**: Verifies backward/adjoint kernel IR is generated
- **CPU vs CUDA difference**: Ensures CPU and CUDA code are distinct
- **CUDA content**: Verifies CUDA-specific constructs (blockDim, threadIdx, etc.)

### GPU Execution Tests (`test_kernels.py`)

These tests require an actual GPU:

- **Arithmetic kernels**: Basic +, -, *, / operations
- **Vector operations**: dot product, cross product, normalize
- **Control flow**: if/else, loops
- **Math functions**: sin, cos, exp, log
- **Backward pass**: Gradient computation

## Expected Output

### Without GPU

```
$ python3 -m pytest tests/cuda/test_extraction.py -v

tests/cuda/test_extraction.py::TestCUDAExtraction::test_forward_extraction[arithmetic] PASSED
tests/cuda/test_extraction.py::TestCUDAExtraction::test_forward_extraction[vector] PASSED
...
tests/cuda/test_extraction.py::TestCUDAExtraction::test_backward_extraction[arithmetic] PASSED
...
========================= X passed in Y seconds =========================
```

### With GPU

```
$ python3 -m pytest tests/cuda/test_kernels.py -v

tests/cuda/test_kernels.py::TestCUDADeviceInfo::test_cuda_device_exists PASSED
tests/cuda/test_kernels.py::TestCUDAKernelExecution::test_simple_arithmetic_cuda PASSED
tests/cuda/test_kernels.py::TestCUDAKernelExecution::test_vector_operations_cuda PASSED
...
========================= X passed in Y seconds =========================
```

## Troubleshooting

### "CUDA not available" error

1. Check CUDA driver is installed: `nvidia-smi`
2. Check CUDA toolkit: `nvcc --version`
3. Verify warp can see CUDA:
   ```python
   import warp as wp
   wp.init()
   print(wp.is_cuda_available())
   ```

### "Out of memory" error

Reduce the test array size in `test_kernels.py`.

### Kernel compilation errors

1. Check CUDA compute capability is supported
2. Update warp: `pip install --upgrade warp-lang`
3. Clear kernel cache: `rm -rf ~/.cache/warp`

## Adding New Tests

1. Add test functions to appropriate test file
2. Use `@pytest.mark.skipif(not CUDA_AVAILABLE, ...)` for GPU-only tests
3. Use `@pytest.mark.parametrize` for testing multiple kernel types
4. Follow existing patterns for kernel compilation and execution
