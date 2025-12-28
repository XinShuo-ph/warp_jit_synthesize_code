# GPU Analysis

## Current CUDA Support
- ir_extractor.py has device param: **Yes** (line 42: `device: str = "cpu"`)
- Tested with device="cuda": **No GPU available** in test environment
- Warp version 1.10.1 supports CUDA when GPU driver is present

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| File extension | `.cpp` | `.cu` |
| Kernel declaration | `void {name}_cpu_kernel_forward(...)` | `extern "C" __global__ void {name}_cuda_kernel_forward(...)` |
| Thread indexing | `task_index` (external loop variable) | `blockDim.x * blockIdx.x + threadIdx.x` |
| Entry point | Regular C function | CUDA `__global__` kernel |
| Loop structure | External loop (caller handles) | Internal grid-stride loop |
| Header | `#define WP_NO_CRT` | `#define __CUDACC__` + debug macros |
| Shared memory | Not used | `wp::tile_shared_storage_t tile_mem` |
| Function decorator | None | `CUDA_CALLABLE` |

## Header Differences

### CPU Header
```cpp
#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

#define builtin_tid1d() wp::tid(task_index, dim)
```

### CUDA Header
```cpp
#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

// Debug support for cuda-gdb
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

#define builtin_tid1d() wp::tid(_idx, dim)
```

## Kernel Structure Differences

### CPU Kernel Template
```cpp
void {name}_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_{name} *_wp_args)
{
    // Direct kernel body execution
    // task_index provided by external launcher
}
```

### CUDA Kernel Template
```cpp
extern "C" __global__ void {name}_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp_args_{name} *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;  // Shared memory for tiles

    // Grid-stride loop for handling large workloads
    for (size_t _idx = blockDim.x * blockIdx.x + threadIdx.x;
         _idx < dim.size;
         _idx += blockDim.x * gridDim.x)
    {
        // Kernel body
    }
}
```

## Changes Needed for GPU Support

1. **Extractor Modification**: The current `ir_extractor.py` already accepts `device` parameter. When loading with `device="cuda"`, warp generates `.cu` files instead of `.cpp`. Need to update file lookup to check for `.cu` extension when `device="cuda"`.

2. **Cache Path Logic**: Update cache file lookup:
   ```python
   if device == "cuda":
       ext = ".cu"
   else:
       ext = ".cpp"
   source_file = os.path.join(module_path, f"{module_dir}{ext}")
   ```

3. **Data Format Extension**: Add `device` field to metadata JSON to track which device the IR was generated for.

4. **Validation**: Need GPU environment to test CUDA code generation paths.

## New GPU-Specific Patterns to Add

- [ ] **Shared Memory Operations**: `wp.tile()`, `wp.block_reduce()`
- [ ] **Atomic Operations**: `wp.atomic_add()`, `wp.atomic_max()`
- [ ] **Warp-Level Primitives**: `wp.warp_shuffle()`, `wp.warp_reduce()`
- [ ] **Grid-Stride Patterns**: Kernels that process more elements than threads
- [ ] **Memory Coalescing Patterns**: Sequential thread access patterns
- [ ] **Texture Memory**: Read-only cached data access

## Code Generation Pipeline Changes

The synthesis pipeline (`code/synthesis/pipeline.py`) would need:

1. **Dual Generation**: Option to generate both CPU and CUDA versions:
   ```python
   def generate_dataset(self, count: int, devices: List[str] = ["cpu"]):
       for device in devices:
           # Generate samples for each device
   ```

2. **Device-Specific Samples**: Some kernels may only make sense on GPU (shared memory, atomics)

3. **Paired Outputs**: Generate matching Python→CPU-IR and Python→CUDA-IR pairs for comparison

## Testing Requirements

To fully validate GPU support:
- Need CUDA-capable GPU with driver
- Install warp with CUDA: `pip install warp-lang[cuda]`
- Test with `wp.set_device("cuda:0")`
- Verify `.cu` file generation in cache
- Compare CPU vs CUDA IR for same kernel

## Estimated Effort

| Task | Effort |
|------|--------|
| Update extractor for .cu files | Low (1-2 hours) |
| Add device tracking to metadata | Low (< 1 hour) |
| GPU-specific kernel patterns | Medium (4-8 hours) |
| Full GPU testing | Requires GPU hardware |
