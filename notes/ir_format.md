# IR Format Documentation

## Overview
Warp's IR is C++ code generated from Python kernels. Each kernel produces
forward and backward (autodiff) functions.

## Structure

### Function Signature
```cpp
void {kernel_name}_{hash}_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_{kernel_name}_{hash} *_wp_args)
```

### Function Body Sections
1. **Argument vars**: Input parameters unpacked from args struct
2. **Primal vars**: Temporary variables for computation
3. **Forward**: Actual computation with SSA form

## Variable Naming Convention
- `var_N`: N-th SSA variable in forward pass
- `var_{param}`: Named parameter (e.g., var_a, var_values)
- Constants: `const wp::float32 var_N = VALUE`

## Operations
Python operators map to wp:: namespace:
- `a + b` → `wp::add(a, b)`
- `a * b` → `wp::mul(a, b)`
- `a[i]` → `wp::address(a, i)` + `wp::load(ptr)`
- `a[i] = v` → `wp::array_store(a, i, v)`

## Source Line Tracking
Comments preserve Python line numbers: `// <L N>`
