# IR Format Documentation

## Structure
```cpp
struct wp_args_{kernel}_{hash} { wp::array_t<T> arg; };  // args struct
void {kernel}_{hash}_cpu_kernel_forward(...);            // forward kernel
void {kernel}_{hash}_cpu_kernel_backward(...);           // backward (autodiff)
WP_API void {kernel}_{hash}_cpu_forward(...);            // entry point
```

## Key IR Patterns
| Python | C++ IR |
|--------|--------|
| `wp.tid()` | `builtin_tid1d()` |
| `a[i]` | `wp::load(wp::address(var_a, var_i))` |
| `a[i] = x` | `wp::array_store(var_a, var_i, var_x)` |
| `a + b` | `wp::add(var_a, var_b)` |
| `for i in range(n)` | `wp::range_t`, loop construct |
| `if cond` | C++ if with `bool var_N` |
| `wp.sin(x)` | `wp::sin(var_x)` |
