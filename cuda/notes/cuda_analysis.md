# CUDA Code Generation Analysis

## Overview
Warp's codegen can generate both CPU (.cpp) and CUDA (.cu) code from the same Python kernel source. The code generation works even without a GPU driver present.

## Key Findings

### Code Generation Works Without GPU
- `builder.codegen('cuda')` successfully generates CUDA code even on CPU-only systems
- No GPU driver required for IR extraction
- This means we can develop and test CUDA IR extraction entirely on this CPU environment

### CPU vs CUDA Code Differences

#### 1. Function Signature
**CPU:**
```cpp
void simple_add_9ad1d227_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_simple_add_9ad1d227 *_wp_args)
```

**CUDA:**
```cpp
extern "C" __global__ void simple_add_9ad1d227_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_a,
    wp::array_t<wp::float32> var_b,
    wp::array_t<wp::float32> var_c)
```

**Differences:**
- CUDA: `extern "C" __global__` decorator
- CUDA: Arguments passed directly, not via struct pointer
- CPU: Uses `task_index` parameter
- CUDA: No `task_index`, uses `_idx` computed from thread/block indices

#### 2. Thread Indexing Macros
**CPU:**
```cpp
#define builtin_tid1d() wp::tid(task_index, dim)
```

**CUDA:**
```cpp
#define builtin_tid1d() wp::tid(_idx, dim)
```

**CUDA has grid-stride loop:**
```cpp
for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
     _idx < dim.size;
     _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
```

#### 3. CUDA-Specific Features
- `__debugbreak()` mapped to `__brkpt()` for cuda-gdb
- Tile shared memory storage: `wp::tile_shared_storage_t tile_mem;`
- Shared memory allocator reset in loop
- Grid-stride loop pattern for thread execution

#### 4. Header Structure
Both include `builtin.h`, but CUDA has additional:
```cpp
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif
```

## Implementation Impact

### For IR Extractor
The existing `ir_extractor.py` already supports `device` parameter:
```python
def extract_ir(kernel, device: str = "cpu", include_backward: bool = True)
```

Simply passing `device="cuda"` should work!

### For Pipeline
The synthesis pipeline needs minimal changes:
1. Pass `device="cuda"` to extraction functions
2. Save `.cu` code instead of `.cpp` code
3. Update metadata to indicate CUDA device

### Testing Strategy
Since CUDA code generation works without GPU:
1. Generate CUDA IR on CPU environment
2. Validate structure and syntax
3. Create test scripts for user to run on actual GPU
4. User can verify compilation and execution on their GPU

## Code Structure Comparison

| Aspect | CPU | CUDA |
|--------|-----|------|
| Function decorator | `void` | `extern "C" __global__ void` |
| Parameter passing | Struct pointer | Direct parameters |
| Thread indexing | `task_index` | `_idx` from blockIdx/threadIdx |
| Execution model | Sequential task loop | Grid-stride parallel loop |
| Shared memory | Not used | `tile_shared_storage_t` |
| File extension | `.cpp` | `.cu` |

## Next Steps
1. Modify `ir_extractor.py` to save device type in metadata (already done!)
2. Modify `pipeline.py` to accept device parameter
3. Create CUDA-specific generator for GPU patterns (atomics, shared memory)
4. Create test suite for GPU validation
