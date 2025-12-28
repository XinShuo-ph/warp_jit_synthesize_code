# CUDA Testing Guide for GPU Validation

## Overview
This guide provides detailed instructions for testing the CUDA backend on actual GPU hardware. While CUDA code generation works on CPU-only systems, this guide covers validation and performance testing on CUDA-capable GPUs.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA Compute Capability 3.5+
  - Recommended: GTX 1060+ / RTX 2060+ / Tesla T4+ / A100
  - Minimum: GTX 750 Ti or equivalent
- **RAM**: 4GB+ system memory
- **Storage**: 1GB for Warp cache and test data

### Software Requirements
1. **NVIDIA Drivers**
   - Version: 450.80.02+ (Linux) / 452.39+ (Windows)
   - Check: `nvidia-smi`

2. **CUDA Toolkit** (optional, Warp bundles its own)
   - Version: 11.0+
   - Check: `nvcc --version`

3. **Python Environment**
   - Python 3.8+
   - warp-lang 1.0+
   - numpy

## Installation on GPU System

### Step 1: Verify GPU
```bash
# Check GPU is detected
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.x    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# ...
```

If `nvidia-smi` fails:
- **Linux**: Install NVIDIA drivers: `sudo apt install nvidia-driver-525`
- **Windows**: Download from https://www.nvidia.com/drivers

### Step 2: Setup Python Environment
```bash
# Create virtual environment (optional but recommended)
python3 -m venv cuda_env
source cuda_env/bin/activate  # Linux/Mac
# cuda_env\Scripts\activate.bat  # Windows

# Install dependencies
pip install warp-lang numpy

# Verify warp sees GPU
python3 -c "import warp as wp; wp.init(); print('Devices:', wp.get_devices())"

# Expected output:
# Devices: ['cpu', 'cuda:0']  # cuda:0 means GPU detected
```

### Step 3: Transfer Project Files
If developing on CPU system, transfer to GPU system:

```bash
# From CPU system:
cd /workspace
tar czf cuda_backend.tar.gz cuda/

# Copy to GPU system:
scp cuda_backend.tar.gz gpu-server:/tmp/

# On GPU system:
cd /tmp
tar xzf cuda_backend.tar.gz
cd cuda
```

## Running Tests

### Quick Test (Automated)
```bash
cd /workspace/cuda  # Or wherever you extracted files
./tests/run_gpu_tests.sh
```

**Expected Output**:
```
======================================================================
GPU Information
======================================================================
name, driver_version, memory.total
NVIDIA GeForce RTX 3080, 525.xx.xx, 10240 MiB

======================================================================
Test 1: CUDA Code Structure Validation
======================================================================
✓ ALL TESTS PASSED
Passed: 6/6

======================================================================
Test 2: GPU Execution Tests
======================================================================
✓ Found 1 CUDA device(s)
  - cuda:0
✓ ALL GPU TESTS PASSED

======================================================================
✓ ALL TESTS PASSED
======================================================================
```

### Manual Test Steps

#### 1. Structure Validation (works on CPU too)
```bash
python3 tests/test_cuda_kernels.py
```

Should show:
- ✓ 6/6 categories passed
- All CUDA structure checks successful

#### 2. GPU Execution Tests
```bash
python3 tests/run_on_gpu.py
```

Should show:
- GPU detected
- Simple arithmetic: PASS
- Vector operations: PASS
- Atomic operations: PASS
- Performance comparison: CPU vs GPU times

## Test Details

### Test 1: Simple Arithmetic
**What**: Adds two arrays element-wise on GPU  
**Size**: 1024 elements  
**Expected**: Max error < 1e-6  
**Purpose**: Verify basic GPU execution

### Test 2: Vector Operations
**What**: Dot product of vec3 arrays  
**Size**: 512 vec3 elements  
**Expected**: Max error < 1e-5  
**Purpose**: Verify vector math on GPU

### Test 3: Atomic Operations
**What**: Parallel reduction using atomic_add  
**Size**: 1000 elements  
**Expected**: Sum matches expected value  
**Purpose**: Verify GPU atomics work correctly

### Test 4: Performance Comparison
**What**: 100 math ops per element  
**Size**: 10,000 elements  
**Expected**: GPU faster than CPU (usually 5-50x)  
**Purpose**: Verify performance benefit

## Interpreting Results

### Performance Expectations

| Workload Size | Expected GPU Speedup | Notes |
|---------------|---------------------|-------|
| < 1K elements | 0.5x - 2x | GPU overhead may dominate |
| 1K - 10K | 2x - 10x | Depends on operation type |
| 10K - 100K | 10x - 30x | Good GPU utilization |
| 100K - 1M | 20x - 50x | Excellent GPU utilization |
| > 1M | 30x - 100x | Best case for GPU |

**Operation types**:
- Simple arithmetic: 5-20x speedup
- Math functions (sin, cos): 20-50x speedup
- Memory-bound: 2-10x speedup

### GPU Memory Usage
```bash
# Monitor during tests
watch -n 1 nvidia-smi

# Look for:
# - Memory usage should be < 100MB for test suite
# - GPU utilization should spike during tests
# - Temperature should remain reasonable (<80°C)
```

## Generating Datasets on GPU

While not required, you can generate on GPU system for validation:

### Generate Test Dataset
```bash
# Small test (fast)
python3 code/synthesis/cuda_pipeline.py -n 100 -o data/gpu_test

# Verify
ls -lh data/gpu_test/
# Should show 100 JSON files

# Check GPU was detected (in output)
# Should NOT show "CUDA driver not found" warning
```

### Generate Large Dataset
```bash
# 10,000 pairs
python3 code/synthesis/cuda_batch_generator.py -n 10000 -o data/gpu_10k

# Monitor progress
tail -f data/gpu_10k/progress.json
```

**Performance**:
- Generation is CPU-bound (compiles kernels)
- GPU presence doesn't speed up generation
- ~10-20 pairs/second typical
- 10K pairs takes ~10-15 minutes

## Troubleshooting

### Issue: "No CUDA devices found"

**Symptoms**:
```
✗ No CUDA devices found!
This test requires a CUDA-capable GPU.
```

**Causes & Solutions**:

1. **No GPU in system**
   ```bash
   lspci | grep -i nvidia
   # Should show NVIDIA GPU
   # If not, check hardware installation
   ```

2. **Drivers not installed**
   ```bash
   nvidia-smi
   # If fails, install drivers:
   # Ubuntu: sudo apt install nvidia-driver-525
   # See: https://www.nvidia.com/drivers
   ```

3. **Drivers loaded but warp can't find**
   ```bash
   # Reinstall warp
   pip uninstall warp-lang
   pip install warp-lang
   
   # Check again
   python3 -c "import warp as wp; wp.init(); print(wp.get_devices())"
   ```

### Issue: "CUDA initialization failed"

**Symptoms**:
```
Warp CUDA initialization failed
```

**Solutions**:

1. **Check CUDA compatibility**
   ```bash
   # Check GPU compute capability
   nvidia-smi --query-gpu=compute_cap --format=csv
   # Should be >= 3.5
   ```

2. **Update drivers**
   ```bash
   # Check driver version
   nvidia-smi | grep "Driver Version"
   # Should be >= 450.80
   ```

3. **Check for conflicts**
   ```bash
   # Unload NVIDIA modules
   sudo rmmod nvidia_uvm
   sudo rmmod nvidia
   # Reload
   sudo modprobe nvidia
   ```

### Issue: GPU slower than CPU

**Symptoms**:
```
CPU Time: 10.5 ms
GPU Time: 15.2 ms
Speedup: 0.69x
⚠ GPU not faster
```

**This is NORMAL for small workloads!**

**Explanation**:
- GPU has launch overhead (~0.1-1ms)
- Small workloads don't fill GPU
- CPU is fast for small tasks

**To see GPU benefits**:
```python
# Modify test in run_on_gpu.py:
n = 1000000  # Increase from 10000
# Re-run test
# Should now show 10-50x speedup
```

### Issue: Out of memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:

1. **Reduce batch size**
   ```python
   # In test file, reduce n:
   n = 1024  # Instead of 1000000
   ```

2. **Check other GPU processes**
   ```bash
   nvidia-smi
   # Kill unnecessary processes
   ```

3. **Use smaller data types**
   ```python
   # float32 instead of float64
   dtype=wp.float32  # 4 bytes
   # dtype=wp.float64  # 8 bytes
   ```

### Issue: Tests hang

**Symptoms**:
- Test starts but never completes
- GPU utilization at 0%

**Solutions**:

1. **Check for deadlock**
   ```bash
   # In another terminal:
   nvidia-smi
   # Look at GPU utilization and memory
   ```

2. **Timeout issue**
   ```python
   # Add timeout to test
   import signal
   signal.alarm(30)  # 30 second timeout
   ```

3. **Driver issue**
   ```bash
   # Reset GPU
   sudo nvidia-smi --gpu-reset
   ```

## Advanced Testing

### Custom Kernel Testing
```python
import warp as wp

wp.init()

@wp.kernel
def my_test_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = a[tid] * 2.0

# Test on GPU
n = 1000
a = wp.array(np.random.rand(n), dtype=wp.float32, device="cuda")
b = wp.zeros(n, dtype=wp.float32, device="cuda")

wp.launch(my_test_kernel, dim=n, inputs=[a, b], device="cuda")
wp.synchronize()

print("Result:", b.numpy()[:10])
```

### Performance Profiling
```bash
# Use NVIDIA Nsight
nvprof python3 tests/run_on_gpu.py

# Or use Nsight Systems
nsys profile python3 tests/run_on_gpu.py
```

### Multi-GPU Testing
```python
# List all GPUs
import warp as wp
wp.init()
devices = wp.get_devices()
cuda_devices = [d for d in devices if 'cuda' in str(d)]

print(f"Found {len(cuda_devices)} GPUs")

# Test on each GPU
for device in cuda_devices:
    print(f"\nTesting on {device}...")
    # Run tests with device=device
```

## Continuous Integration

### Example GitHub Actions (requires GPU runner)
```yaml
name: GPU Tests

on: [push, pull_request]

jobs:
  test-gpu:
    runs-on: [self-hosted, gpu]
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        run: |
          python3 -m pip install warp-lang numpy
      
      - name: Check GPU
        run: nvidia-smi
      
      - name: Run GPU Tests
        run: |
          cd cuda
          ./tests/run_gpu_tests.sh
```

## Next Steps After Tests Pass

1. **Generate production dataset**:
   ```bash
   python3 code/synthesis/cuda_batch_generator.py -n 100000 -o data/production
   ```

2. **Benchmark specific kernels**:
   - Modify `run_on_gpu.py` with your kernel types
   - Measure performance on your target GPU

3. **Deploy to training pipeline**:
   - Use generated JSON pairs
   - Feed to LLM training infrastructure

## Support

If tests fail after following this guide:

1. **Check system logs**: `dmesg | grep -i nvidia`
2. **Verify CUDA samples work**: Run NVIDIA CUDA samples if installed
3. **Check Warp GitHub issues**: https://github.com/NVIDIA/warp/issues
4. **Provide details**:
   - GPU model (`nvidia-smi`)
   - Driver version
   - warp version (`pip show warp-lang`)
   - Complete error message

---

**Success Criteria**: All tests pass, GPU detected, reasonable performance observed.
