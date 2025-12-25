# IR Format Documentation

## Structure

Warp generates C++ code in consistent structure:

### 1. Header & Definitions
- Includes: `builtin.h` (warp runtime)
- Macros: Type casts, thread ID helpers
- Block dim: 1 for CPU, configurable for GPU

### 2. Argument Struct
```cpp
struct wp_args_<kernel_name>_<hash> {
    wp::array_t<type> param1;
    ...
};
```

### 3. Forward Kernel
```cpp
void <kernel>_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_* _wp_args)
```
Contains: argument loading, primal vars, computation

### 4. Backward Kernel (autodiff)
Similar to forward but includes adjoint variables

### 5. Entry Points
External C functions that iterate over task indices

## Variable Naming

- `var_N`: Primal variable (N is sequence number)
- `adj_N`: Adjoint variable for autodiff
- Comments preserve Python line numbers: `<L 14>`

## Type Mapping

Python → C++:
- `float` → `wp::float32`
- `int` → `wp::int32`
- `wp.vec3` → `wp::vec3` (struct)
- `wp.array` → `wp::array_t<T>`
