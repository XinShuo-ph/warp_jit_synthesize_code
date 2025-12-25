# Warp IR Format

## Generated C++ Structure
```cpp
struct wp_args_kernel_HASH { /* typed arguments */ };

void kernel_HASH_cpu_kernel_forward(dim, task_index, args);
void kernel_HASH_cpu_kernel_backward(dim, task_index, args, adj_args);

extern "C" WP_API void kernel_HASH_cpu_forward(dim, args);  // entry point
extern "C" WP_API void kernel_HASH_cpu_backward(dim, args, adj_args);
```

## IR Sections (within each function)
1. **argument vars** - Unpack args struct to local variables
2. **primal vars** - Declare intermediate computation variables  
3. **dual vars** (backward only) - Adjoint variables for autodiff
4. **forward** - Main computation with source line comments
5. **reverse** (backward only) - Adjoint computations in reverse order

## Type Mappings
| Python | C++ IR |
|--------|--------|
| `float` | `wp::float32` |
| `int` | `wp::int32` |
| `wp.vec3` | `wp::vec_t<3, wp::float32>` |
| `wp.array(dtype=T)` | `wp::array_t<T>` |

## Key Operations
- `wp::tid()` → thread index
- `wp::address()` → pointer to array element
- `wp::load()/store()` → array access
- `wp::add/mul/sub()` → arithmetic with adjoint support
