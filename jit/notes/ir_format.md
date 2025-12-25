# Warp IR (C++ Code) Format

## Structure

Generated C++ consists of:
1. **Args struct**: `wp_args_{kernel_name}` with typed members for each kernel arg
2. **Forward function**: `{name}_cpu_kernel_forward()` - computes primal values
3. **Backward function**: `{name}_cpu_kernel_backward()` - computes gradients
4. **Entry points**: C-exported wrappers for Python FFI

## Forward Function Layout

```cpp
void kernel_forward(wp::launch_bounds_t dim, size_t task_index, wp_args *args) {
    // 1. Argument vars - unpack from struct
    // 2. Primal vars - declare temporaries
    // 3. Forward pass - SSA form operations with line comments
}
```

## Key Patterns

- `wp::tid()` → `builtin_tid1d()` thread index
- `a[i]` → `wp::address()` + `wp::load()` / `wp::array_store()`
- `a + b` → `wp::add(a, b)`
- `a * b` → `wp::mul(a, b)`
- Branches → standard C++ if/else
- Loops → standard C++ for loops

## Backward (Adjoint) Code

- Forward pass re-run to reconstruct primal values
- Reverse pass applies chain rule: `adj_{op}(primal, adj_in, adj_out)`
- Gradient arrays: `adj_a`, `adj_b`, etc.

## Line Mapping

Comments `<L N>` map to Python source line N for debugging.
