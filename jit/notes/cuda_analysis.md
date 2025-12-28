# CPU vs CUDA Code Generation Analysis

## Overview

Warp's code generation uses the same kernel definitions but produces different C++ code for CPU and CUDA backends. The `ir_extractor.py` already supports a `device` parameter that controls which backend code is generated.

## Key Differences

### 1. Function Naming
- CPU: `{kernel_name}_{hash}_cpu_kernel_forward`
- CUDA: `{kernel_name}_{hash}_cuda_kernel_forward`

### 2. Function Signature

**CPU:**
```cpp
void kernel_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernel *_wp_args)
```

**CUDA:**
```cpp
void kernel_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<T> arg1,
    wp::array_t<T> arg2,
    ...)
```

### 3. Thread Execution Model

**CPU:** Single-threaded sequential execution per task
```cpp
var_0 = builtin_tid1d();
// ... rest of kernel body
```

**CUDA:** Grid-stride loop with CUDA thread indexing
```cpp
for (size_t _idx = blockDim.x * blockIdx.x + threadIdx.x;
     _idx < dim.size;
     _idx += blockDim.x * gridDim.x) {
    // ... kernel body
}
```

### 4. Shared Memory

CUDA kernels include shared memory initialization:
```cpp
wp::tile_shared_storage_t tile_mem;
wp::tile_shared_storage_t::init();
```

### 5. Same Internal Functions

Both backends use identical `wp::` namespace functions:
- `wp::address()` - array element addressing
- `wp::load()` - memory load
- `wp::add()`, `wp::mul()`, etc. - math operations
- `wp::array_store()` - array element store
- `wp::atomic_add()`, `wp::atomic_max()` - atomics
- `wp::dot()`, `wp::cross()` - vector operations

## Implementation Impact

### What Needs to Change

1. **ir_extractor.py**: Already supports `device` parameter ✓
2. **pipeline.py**: Add `--device` CLI argument
3. **batch_generator.py**: Add device parameter to batch functions
4. **generator.py**: No changes needed (kernels are device-agnostic)
5. **Test scripts**: Create CUDA-specific tests for user to run on GPU

### What Stays the Same

- Kernel source code (Python) is identical for CPU and CUDA
- Kernel generation logic in generator.py
- JSON output format (just add `device: "cuda"` to metadata)

## Test Commands for User

```bash
# Check CUDA availability
python3 -c "import warp as wp; wp.init(); print('CUDA:', wp.is_cuda_available())"

# Generate CUDA IR samples
python3 code/synthesis/pipeline.py --device cuda -n 10 -o data/cuda/

# Compare CPU vs CUDA output
diff data/cpu/synth_0000.json data/cuda/synth_0000.json
```
