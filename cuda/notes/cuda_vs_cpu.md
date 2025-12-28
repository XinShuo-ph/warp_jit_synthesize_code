# CUDA vs CPU Code Differences

## Overview

Warp generates different code for CPU and CUDA backends. The core algorithm is the same, but the execution model differs.

## Key Differences

### 1. Thread Indexing

**CPU (sequential execution):**
```cpp
void kernel_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernel *_wp_args)
{
    // task_index is the thread ID
    var_0 = builtin_tid1d();
    ...
}
```

**CUDA (parallel execution):**
```cpp
void kernel_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<...> var_a,
    ...)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        ...
    }
}
```

### 2. Shared Memory

CUDA kernels include shared memory initialization:
```cpp
wp::tile_shared_storage_t tile_mem;
// reset shared memory allocator
wp::tile_shared_storage_t::init();
```

CPU kernels don't have this.

### 3. Function Signatures

**CPU:** Uses pointer to args struct
```cpp
void kernel_cpu_kernel_forward(..., wp_args_kernel *_wp_args)
```

**CUDA:** Uses expanded arguments directly
```cpp
void kernel_cuda_kernel_forward(..., wp::array_t<float> var_a, ...)
```

### 4. Suffix Naming

- CPU: `*_cpu_kernel_forward`, `*_cpu_kernel_backward`
- CUDA: `*_cuda_kernel_forward`, `*_cuda_kernel_backward`

## What's the Same

1. **Variable naming**: Same `var_0`, `var_1`, etc. pattern
2. **Operations**: Same `wp::add`, `wp::mul`, `wp::sin`, etc.
3. **Adjoint structure**: Same backward pass algorithm
4. **Memory access**: Same `wp::address`, `wp::load`, `wp::array_store`

## Code Generation Details

The `builder.codegen(device)` function in `warp._src.context.ModuleBuilder` handles the device-specific code generation:

```python
builder = warp._src.context.ModuleBuilder(module, options, hasher)
cpu_code = builder.codegen("cpu")    # Generates .cpp
cuda_code = builder.codegen("cuda")  # Generates .cu
```

Both can be called without a GPU present - the code generation is pure Python.

## Backward Pass

Both CPU and CUDA backward passes:
1. Re-compute forward pass values
2. Apply reverse-mode autodiff
3. Use `adj_*` prefix for adjoint variables
4. Call `wp::adj_*` functions for gradient propagation

Example adjoint operations:
```cpp
wp::adj_sin(var_13, adj_13, adj_14);
wp::adj_add(var_4, var_7, adj_4, adj_7, adj_8);
```
