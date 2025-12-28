# GPU Analysis

## Current CUDA Support
- ir_extractor.py has device param: **Yes** (`device="cpu"` or `device="cuda"`)
- Tested with device="cuda": **No GPU available** - environment is CPU-only
- Code is ready for CUDA but requires GPU hardware to generate `.cu` files

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| **Entry point** | `WP_API void {name}_cpu_forward(...)` | `extern "C" __global__ void {name}_cuda_kernel_forward(...)` |
| **Kernel decorator** | Regular C++ function | `__global__` CUDA kernel |
| **Thread indexing** | Sequential `for` loop over `task_index` | `blockDim.x * blockIdx.x + threadIdx.x` |
| **Parallelism model** | Single-threaded iteration | Massively parallel grid/block |
| **Memory** | Regular CPU memory | Device memory (`cudaMalloc`) |
| **Shared memory** | Not used | `wp::tile_shared_storage_t tile_mem` |
| **Synchronization** | Not needed | `__syncthreads()` for tile operations |
| **File extension** | `.cpp` | `.cu` |

## Kernel Template Differences

### CPU Forward Kernel
```cpp
void {name}_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_{name} *_wp_args)
{
    // Sequential execution, task_index is loop variable
}

// Entry point
WP_API void {name}_cpu_forward(wp::launch_bounds_t dim, wp_args_{name} *_wp_args)
{
    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        {name}_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}
```

### CUDA Forward Kernel
```cuda
extern "C" __global__ void {name}_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp_args_{name} *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
    
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // Parallel execution across GPU threads
    }
}
```

## Changes Needed for GPU Support in This Branch

1. **Hardware requirement**: Need CUDA-capable GPU and NVIDIA driver installed
2. **Warp initialization**: Already handles CUDA devices automatically when available
3. **ir_extractor.py changes**: None needed - already supports `device="cuda"`, looks for `.cu` files
4. **pipeline.py changes**: Minor - add device parameter to synthesis pipeline
5. **Data format**: Add `device` field (already present in data schema)

## Implementation Status

| Component | CPU Support | CUDA Ready |
|-----------|-------------|------------|
| `ir_extractor.py` | ✅ Working | ✅ Has device param, untested |
| `generator.py` | ✅ Working | ✅ Kernels are device-agnostic |
| `pipeline.py` | ✅ Working | ⚠️ Needs device parameter passthrough |
| `batch_generator.py` | ✅ Working | ⚠️ Needs device parameter passthrough |

## New GPU-Specific Patterns to Add

When GPU is available, the following patterns could be added to the generator:

- [ ] **Tile operations**: `wp.tile_load()`, `wp.tile_store()`, tile matmul
- [ ] **Atomic operations**: `wp.atomic_add()`, `wp.atomic_sub()`
- [ ] **Shared memory**: Tile-based algorithms requiring shared storage
- [ ] **Warp-level primitives**: `wp.warp_shuffle()`, warp reductions
- [ ] **Memory coalescing patterns**: Strided vs coalesced access patterns

## CUDA-Specific IR Patterns (Expected)

| Python | CUDA IR |
|--------|---------|
| `wp.tid()` | `blockDim.x * blockIdx.x + threadIdx.x` via `builtin_tid1d()` |
| `wp.tile_load(a, i)` | Loads using shared memory with `__syncthreads()` |
| `wp.atomic_add(a, i, v)` | `atomicAdd()` CUDA intrinsic |
| Tile operations | Uses `tile_shared_storage_t` for inter-thread communication |

## Verification Plan (When GPU Available)

1. Run `python3 code/extraction/ir_extractor.py` with `device="cuda"`
2. Compare generated `.cu` vs `.cpp` for same kernel
3. Verify all kernel types generate valid CUDA code
4. Run batch generation with `--device cuda`
5. Validate CUDA IR structure matches expected patterns

## Recommendations

1. **For training data**: Generate both CPU and CUDA pairs for each kernel to learn device-specific patterns
2. **For testing**: Add CI/CD with GPU runner to validate CUDA path
3. **For scaling**: CUDA compilation is typically faster due to NVCC caching; batch generation may be more efficient on GPU systems
