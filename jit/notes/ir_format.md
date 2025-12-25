# Warp IR Format

## Overview
Warp's Intermediate Representation (IR) is a linear sequence of C++-like statements, organized into Basic Blocks. It is generated during the `Adjoint` pass of compilation.

## Structure
The extracted IR consists of a list of lines, representing the `body_forward` of each `Block` in the kernel's Control Flow Graph (CFG).

```cpp
// Block 0
var_0 = builtin_tid1d();
var_1 = wp::address(var_x, var_0);
var_2 = wp::load(var_1);
...
```

## Key Characteristics

1.  **SSA Form**: Variables are assigned once (mostly) and named `var_N`.
2.  **Builtins**: Operations map to C++ functions in the `wp::` namespace (e.g., `wp::add`, `wp::mul`, `wp::load`) or global builtins (`builtin_tid1d`).
3.  **Memory Access**:
    -   `wp::address(arr, idx)`: Computes address.
    -   `wp::load(ptr)`: Reads value.
    -   `wp::array_store(arr, idx, val)`: Writes value.
4.  **Control Flow**:
    -   Represented by multiple blocks.
    -   Conditionals and Loops involve branching between blocks (though the raw statement list for a single block is linear).
    -   (Note: Deep inspection of control flow requires analyzing `Adjoint.blocks` linkage, but the linear text representation usually captures the sequential logic within blocks).

## Data Types
-   Scalars: `float`, `int`, etc.
-   Vectors/Matrices: `wp::vec3`, `wp::mat33`.
-   Arrays: Passed as `var_name` handles.

## Use Case for Training
This IR provides a simplified, low-level view of the computation, suitable for learning the mapping between Python syntax and vectorized/GPU-ready operations.
