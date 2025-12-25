# Warp C++ IR Format

## Function Structure
```
void <kernel_name>_<hash>_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_<kernel_name>_<hash> *_wp_args)
```

## Variable Sections
1. **Argument vars**: Parameters from args struct
2. **Primal vars**: Intermediate computations
3. **Forward**: Actual computation with Python source comments

## Variable Naming
- `var_N`: Sequential numbering (var_0, var_1, ...)
- var_0: Usually thread ID from wp.tid()
- Constants: `const wp::float32 var_2 = 2.0`

## Operation Mapping
| Python | C++ IR |
|--------|--------|
| `wp.tid()` | `builtin_tid1d()` |
| `arr[idx]` | `wp::address(arr, idx)` + `wp::load(ptr)` |
| `a * b` | `wp::mul(a, b)` |
| `a + b` | `wp::add(a, b)` |
| `arr[idx] = val` | `wp::array_store(arr, idx, val)` |

## Key Observations
- Each operation gets own variable (SSA-like)
- Python code preserved in comments with line numbers
- Forward and backward passes generated
