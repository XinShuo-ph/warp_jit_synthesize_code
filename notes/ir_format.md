# Warp C++ IR Format

## Structure
Generated C++ follows a consistent pattern:

### Function Signature
```
void <kernel_name>_<hash>_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_<kernel_name>_<hash> *_wp_args)
```

### Variable Sections
1. **Argument vars**: Function parameters unpacked from args struct
2. **Primal vars**: Intermediate computation variables
3. **Forward**: Actual computation with Python source as comments

## Variable Naming
- `var_N`: Variables numbered sequentially (var_0, var_1, ...)
- var_0: Usually thread ID from `wp.tid()`
- Pointers end with `*` (e.g., `wp::float32* var_1`)
- Constants are `const` (e.g., `const wp::float32 var_2 = 2.0`)

## Operation Mapping
| Python | C++ IR |
|--------|--------|
| `wp.tid()` | `builtin_tid1d()` |
| `arr[idx]` | `wp::address(arr, idx)` then `wp::load(ptr)` |
| `a * b` | `wp::mul(a, b)` |
| `a + b` | `wp::add(a, b)` |
| `a - b` | `wp::sub(a, b)` |
| `arr[idx] = val` | `wp::array_store(arr, idx, val)` |
| `wp.sqrt(x)` | `wp::sqrt(x)` |
| `wp.length(v)` | `wp::length(v)` |

## Key Observations
- Each operation gets its own variable assignment
- SSA-like form (each variable assigned once)
- Original Python code preserved in comments with line numbers
- Both forward and backward (autodiff) passes generated
