# GPU IR Format Analysis

## Key Differences: CPU vs CUDA

### Function Signature

**CPU:**
```cpp
void kernel_name_hash_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernel_name_hash *_wp_args)
```

**CUDA:**
```cpp
extern "C" __global__ void kernel_name_hash_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    [direct array arguments...])
```

Key differences:
- CUDA has `extern "C" __global__` qualifier (marks as CUDA kernel)
- CUDA passes arrays directly as arguments
- CPU uses struct pointer `_wp_args` for arguments
- CPU has `task_index` for sequential execution
- CUDA uses thread grid for parallel execution

### Thread Indexing

**CPU:**
```cpp
#define builtin_tid1d() wp::tid(task_index, dim)
var_0 = builtin_tid1d();
```

**CUDA:**
```cpp
#define builtin_tid1d() wp::tid(_idx, dim)

for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
     _idx < dim.size;
     _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
{
    var_0 = builtin_tid1d();
    // ... kernel body
}
```

CUDA features:
- `blockDim.x` - threads per block
- `blockIdx.x` - block index in grid
- `threadIdx.x` - thread index within block
- `gridDim.x` - total number of blocks
- Grid-stride loop: `_idx += blockDim.x * gridDim.x`

### Memory Management

**CUDA:**
```cpp
wp::tile_shared_storage_t tile_mem;  // Shared memory support
wp::tile_shared_storage_t::init();   // Reset shared memory allocator
```

This provides infrastructure for CUDA shared memory (`__shared__`) operations.

### Atomic Operations

Both CPU and CUDA use same warp API:
```cpp
var_3 = wp::atomic_max(var_result, var_1, var_4);
```

Warp handles the platform differences internally.

## CUDA-Specific Patterns Detected

From generated samples:
1. ✓ `__global__` kernel qualifier
2. ✓ `blockDim.x`, `blockIdx.x`, `threadIdx.x`, `gridDim.x`
3. ✓ Grid-stride loop pattern
4. ✓ Shared memory infrastructure (`tile_mem`)
5. ✓ Atomic operations work correctly

## IR Output Structure

Both CPU and CUDA generate complete, compilable code:

```
[Preprocessor directives and includes]
#define WP_TILE_BLOCK_DIM 256
#include "builtin.h"

[Helper macros]
#define builtin_tid1d() ...

[Forward kernel function]
void/extern "C" __global__ void kernel_name_hash_{device}_kernel_forward(...) { ... }

[Backward kernel function (if enabled)]
void/extern "C" __global__ void kernel_name_hash_{device}_kernel_backward(...) { ... }
```

## Test Results

Generated CUDA IR for all 6 kernel categories:
- ✓ arithmetic (2 samples)
- ✓ vector (6 samples)  
- ✓ matrix (1 sample)
- ✓ control_flow (1 sample)
- ✓ math (2 samples)
- ✓ atomic (3 samples)

All samples contain proper CUDA patterns and compile-ready code.

## Next Steps

The IR extraction now works for both CPU and CUDA. The current generator already creates valid kernels that compile to CUDA. However, we could enhance the generator to use more GPU-specific patterns:

1. Explicit shared memory usage (`wp.shared_array()`)
2. Synchronization primitives (`wp.syncthreads()`)
3. More complex atomic patterns
4. Thread-cooperative algorithms

These enhancements would be done in M3 (forward pass improvements) and M4 (backward pass).
