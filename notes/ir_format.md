# Warp IR Format

The Intermediate Representation (IR) extracted from Warp kernels is C++ code designed to be compiled by a backend compiler (nvcc/gcc/clang). This note describes its structure.

## 1. Argument Struct
Each kernel defines a struct to hold its arguments. The struct name includes the kernel name and a hash.

```cpp
struct wp_args_kernelname_hash {
    wp::array_t<wp::float32> arg1;
    wp::int32 arg2;
    // ...
};
```

## 2. Forward Function
The main execution logic is in a function suffixed with `_cpu_kernel_forward` (or `_cuda_kernel_forward`).

```cpp
void kernelname_hash_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernelname_hash *_wp_args)
{
    // ... body ...
}
```

### Body Structure
1.  **Argument Unpacking**: Arguments are unpacked from `_wp_args` into local variables prefixed with `var_`.
    ```cpp
    wp::array_t<wp::float32> var_arg1 = _wp_args->arg1;
    ```
2.  **Primal Variables**: Local variables (SSA form) are declared at the top, named `var_0`, `var_1`, etc.
    ```cpp
    wp::int32 var_0;
    wp::float32 var_1;
    ```
3.  **Logic**: The kernel logic follows. Operations are typically Warp built-ins (`wp::add`, `wp::load`, `wp::array_store`) or standard C++ constructs (`if`, `goto`).
    *   Loops are often implemented using `goto` and labels (`start_for_0`, `end_for_0`).
    *   Comments often indicate the corresponding Python source line: `// c[tid] = a[tid] + b[tid] <L 10>`

## 3. Backward Function (Adjoint)
If `enable_backward=True` (default), a `_cpu_kernel_backward` function is generated.

```cpp
void kernelname_hash_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernelname_hash *_wp_args,
    wp_args_kernelname_hash *_wp_adj_args)
{
    // ... body ...
}
```

### Body Structure
1.  **Argument Unpacking**: Similar to forward, but also unpacks adjoint arguments (`adj_arg1`) from `_wp_adj_args`.
2.  **Dual Variables**: Adjoint variables corresponding to primal variables are declared (`adj_0`, `adj_1`).
3.  **Forward Replay**: Parts of the forward pass may be replayed to recompute necessary values.
4.  **Reverse Pass**: The adjoint logic executes in reverse order, accumulating gradients using `wp::adj_add`, `wp::adj_mul`, etc.

## 4. Key Observations for Synthesis
*   **Variable Naming**: Internal variables are numbered sequentially (`var_0`, `var_1`). Argument variables preserve names (`var_x`).
*   **Control Flow**: Python `for` loops become `goto` based loops in C++. Python `if` becomes C++ `if`.
*   **Type System**: Warp types map to C++ types (`wp::array_t`, `wp::float32`).
