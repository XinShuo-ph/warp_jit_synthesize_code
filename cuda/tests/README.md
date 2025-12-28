# CUDA Backend Test Suite

## Overview
This directory contains tests for validating the CUDA backend implementation. Tests are divided into two categories:
1. **Structure validation** (runs on CPU-only systems)
2. **GPU execution** (requires CUDA-capable GPU)

## Test Files

### test_cuda_kernels.py
**Purpose**: Validate CUDA code generation and structure  
**Requirements**: Python 3, warp-lang (no GPU needed)  
**What it tests**:
- CUDA code contains proper `extern "C" __global__` decorators
- Grid-stride loop patterns present
- All kernel categories generate valid CUDA
- CPU vs CUDA code differences

**Run**:
```bash
python3 tests/test_cuda_kernels.py
```

**Expected output**:
```
✓ ALL TESTS PASSED
Passed: 6/6 categories
```

---

### run_on_gpu.py
**Purpose**: Execute CUDA kernels on actual GPU hardware  
**Requirements**: CUDA-capable GPU, NVIDIA drivers, warp-lang  
**What it tests**:
- GPU availability detection
- Simple arithmetic operations
- Vector operations (wp.vec3)
- Atomic operations
- CPU vs GPU performance comparison

**Run**:
```bash
python3 tests/run_on_gpu.py
```

**Expected output** (on GPU system):
```
✓ Found 1 CUDA device(s)
✓ ALL GPU TESTS PASSED
Speedup: X.XXx
```

**Expected output** (on CPU-only system):
```
✗ No CUDA devices found!
Troubleshooting:
  1. Check GPU: nvidia-smi
  2. Check CUDA drivers installed
  ...
```

---

### run_gpu_tests.sh
**Purpose**: Comprehensive test runner script  
**Requirements**: CUDA-capable GPU, bash  
**What it does**:
1. Checks for nvidia-smi and GPU
2. Verifies Python and warp installation
3. Runs structure validation tests
4. Runs GPU execution tests
5. Provides next steps

**Run**:
```bash
./tests/run_gpu_tests.sh
```

Or:
```bash
bash tests/run_gpu_tests.sh
```

---

## Running Tests on GPU

### Prerequisites
1. **CUDA-capable GPU**
   - Check: `nvidia-smi`
   - Should show GPU name and driver version

2. **NVIDIA Drivers**
   - CUDA 11.0+ recommended
   - Check: `nvidia-smi` should work without errors

3. **Python Environment**
   ```bash
   pip install warp-lang numpy
   ```

### Quick Start
```bash
# On GPU system:
cd /workspace/cuda
./tests/run_gpu_tests.sh
```

### Manual Testing
```bash
# Structure tests (works on CPU):
python3 tests/test_cuda_kernels.py

# GPU execution (requires GPU):
python3 tests/run_on_gpu.py
```

---

## Understanding Test Results

### Structure Tests
These validate that generated CUDA code has correct syntax and structure:
- ✓ `extern "C" __global__` present
- ✓ Grid-stride loop pattern
- ✓ Thread indexing (blockIdx, threadIdx)
- ✓ Shared memory declarations

**These tests pass on CPU-only systems** because they only check code structure, not execution.

### GPU Execution Tests
These compile and run kernels on GPU:
- Test correctness (results match expected values)
- Test different kernel types (arithmetic, vector, atomic)
- Measure performance vs CPU

**These tests require actual GPU** and will fail on CPU-only systems.

---

## Troubleshooting

### "No CUDA devices found"
**Cause**: No GPU or drivers not installed  
**Solution**:
1. Check GPU exists: `lspci | grep -i nvidia`
2. Check drivers: `nvidia-smi`
3. Install drivers if needed
4. Reinstall warp: `pip install --upgrade warp-lang`

### "CUDA driver initialization failed"
**Cause**: Driver version mismatch or not loaded  
**Solution**:
1. Check driver loaded: `nvidia-smi`
2. Check CUDA version: `nvcc --version`
3. Update drivers if needed

### Tests pass but GPU slower than CPU
**Cause**: Small workload, GPU overhead dominates  
**Expected**: Normal for small test cases  
**Solution**: Not an error - GPU benefits show on larger workloads

### Import errors
**Cause**: Missing dependencies or wrong Python path  
**Solution**:
```bash
pip install warp-lang numpy
# Make sure you're in the cuda/ directory when running tests
cd /workspace/cuda
python3 tests/test_cuda_kernels.py
```

---

## Test Data

### Structure Test Data
Generated in: `/tmp/cuda_test_*`  
- Temporary files, cleaned up automatically
- Each test run generates fresh data

### GPU Test Data
Generated in: `/tmp/warp_cuda_synthesis/`  
- Temporary compiled kernels
- Can be deleted safely

---

## Performance Benchmarks

Expected performance characteristics:

| Test | Array Size | CPU Time | GPU Time | Expected Speedup |
|------|------------|----------|----------|------------------|
| Simple Add | 1K | ~0.1ms | ~0.1ms | ~1x (overhead) |
| Simple Add | 1M | ~5ms | ~0.5ms | ~10x |
| Vector Ops | 10K | ~2ms | ~0.3ms | ~5-10x |
| Math Heavy | 10K | ~50ms | ~2ms | ~20-30x |

**Note**: Actual performance depends on GPU model, CPU model, and system configuration.

---

## Adding New Tests

### Structure Test Template
```python
def test_my_feature():
    """Test description."""
    print("Testing my feature...")
    
    # Generate kernels
    pairs = run_cuda_pipeline(n=5, ...)
    
    # Validate structure
    for pair in pairs:
        assert "expected_pattern" in pair["cuda_forward"]
    
    print("✓ Test passed")
    return True
```

### GPU Execution Test Template
```python
def test_my_gpu_feature():
    """GPU test description."""
    
    @wp.kernel
    def my_kernel(...):
        # Kernel code
        pass
    
    # Create GPU arrays
    a_gpu = wp.array(..., device="cuda")
    
    # Launch
    wp.launch(my_kernel, dim=n, inputs=[a_gpu], device="cuda")
    wp.synchronize()
    
    # Verify results
    result = a_gpu.numpy()
    assert np.allclose(result, expected)
```

---

## CI/CD Integration

### CPU-only CI (GitHub Actions, etc.)
```yaml
- name: Test CUDA Code Generation
  run: |
    pip install warp-lang
    python3 tests/test_cuda_kernels.py
```

### GPU CI (requires GPU runners)
```yaml
- name: Test GPU Execution
  run: |
    nvidia-smi
    ./tests/run_gpu_tests.sh
```

---

## Next Steps After Tests Pass

1. **Generate larger dataset**:
   ```bash
   python3 code/synthesis/cuda_batch_generator.py -n 10000
   ```

2. **Analyze generated data**:
   ```bash
   ls -lh data/cuda_large/
   python3 -c "import json; print(json.load(open('data/cuda_large/cuda_synth_0000.json'))['metadata'])"
   ```

3. **Use for LLM training**:
   - All pairs in `data/cuda_large/*.json`
   - Format: `{"python_source": "...", "cuda_forward": "...", "metadata": {...}}`
   - Ready for training data pipelines
