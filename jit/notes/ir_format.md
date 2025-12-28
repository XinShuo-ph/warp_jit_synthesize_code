# Warp IR Format Documentation

## Generated C++ Structure
Each kernel produces forward and backward (adjoint) C++ functions.

## Forward Kernel Pattern
```cpp
void {kernel}_{hash}_{device}_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_{kernel}_{hash} *_wp_args)
{
    // argument vars - unpacked from args struct
    wp::array_t<T> var_name = _wp_args->name;
    
    // primal vars - intermediate computation variables
    wp::int32 var_0;  // named var_0, var_1, etc.
    
    // forward - actual computation with line comments from Python
    var_0 = builtin_tid1d();
    var_1 = wp::address(var_arr, var_0);
    // ... operations map directly to Python lines
}
```

## Key Patterns
- `var_N`: Intermediate variables (SSA-like form)
- `wp::address(arr, idx)`: Array element pointer
- `wp::load(ptr)`: Load value from pointer
- `wp::array_store(arr, idx, val)`: Store to array
- `wp::add/mul/...`: Arithmetic operations
- Line comments show original Python source

## Backward Kernel
Contains both forward replay and reverse-mode autodiff:
- `adj_N`: Adjoint (gradient) of `var_N`
- `wp::adj_*`: Adjoint operations for each forward op
