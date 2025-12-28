# GPU Analysis

## Current CUDA Support
- `ir_extractor.py` has device param: **Yes** (`extract_ir(kernel, device="cpu"|"cuda")`)
- Tested with `device="cuda"`: **no GPU** (environment lacks NVIDIA CUDA driver)
- Code should work with CUDA if GPU available (same `ModuleBuilder.codegen()` API)

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| **Header** | `#define WP_NO_CRT` | `#define WP_NO_CRT` + CUDA breakpoint macro |
| **Thread ID macro** | `wp::tid(task_index, dim)` | `wp::tid(_idx, dim)` |
| **Kernel declaration** | `void {name}_cpu_kernel_forward(...)` | `extern "C" __global__ void {name}_cuda_kernel_forward(...)` |
| **Arguments** | Passed via `wp_args_{name}` struct pointer | Passed as individual function parameters |
| **Parallelization** | Serial `for` loop over `task_index` in wrapper | Grid-stride loop with `blockDim/blockIdx/threadIdx` |
| **Entry points** | `{name}_cpu_forward()` with `extern "C"` | `{name}_cuda_kernel_forward()` with `__global__` |
| **Shared memory** | Stack-allocated tile storage | `wp::tile_shared_storage_t::init()` per iteration |
| **Indentation** | 4 spaces | 8 spaces (nested in grid-stride loop) |

## Key Structural Differences

### CPU Kernel Pattern
```cpp
// Args passed via struct
struct wp_args_k_add {
    wp::array_t<wp::float32> a;
    wp::array_t<wp::float32> b;
};

void k_add_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,          // ← Sequential iteration
    wp_args_k_add *_wp_args)
{
    // Extract args from struct
    wp::array_t<wp::float32> var_a = _wp_args->a;
    // ... kernel body
}

// Wrapper loops over all tasks
WP_API void k_add_cpu_forward(...) {
    for (size_t task_index = 0; task_index < dim.size; ++task_index) {
        k_add_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}
```

### GPU Kernel Pattern
```cpp
extern "C" __global__ void k_add_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_a,  // ← Direct params
    wp::array_t<wp::float32> var_b,
    wp::array_t<wp::float32> var_out)
{
    // Grid-stride loop for large workloads
    for (size_t _idx = blockDim.x * blockIdx.x + threadIdx.x;
         _idx < dim.size;
         _idx += blockDim.x * gridDim.x)
    {
        // ... kernel body using _idx
    }
}
```

## Changes Needed for GPU Training Data

1. **No code changes required**: `ir_extractor.py` already supports `device="cuda"` parameter
2. **Environment requirement**: Need NVIDIA GPU + CUDA driver to test
3. **Data format**: Same JSONL format, just `"codegen_device": "cuda"` and `.cu` source

## New GPU-Specific Patterns to Add (for synthesis)

- [ ] `__global__` kernel declaration syntax
- [ ] Grid-stride loop pattern (`blockDim.x * blockIdx.x + threadIdx.x`)
- [ ] Direct parameter passing (no struct indirection)
- [ ] `__syncthreads()` for shared memory operations
- [ ] Warp-level primitives (`__shfl_*`, `__ballot_sync`)
- [ ] Memory coalescing patterns for optimal access
- [ ] Tile operations with `wp::tile_shared_storage_t::init()`

## Recommendations

1. **Immediate**: Run `m2_generate_pairs.py` on a GPU-enabled machine to capture CUDA samples
2. **Medium-term**: Add GPU-specific kernel examples (matmul, stencil, prefix sum)
3. **Long-term**: Build CPU/GPU parallel corpus for transfer learning
