# Extracted IR Format (C++ Source)

The IR extracted from Warp is the generated C++ source code that Warp compiles into kernels. This representation is closer to the final machine code than the Python AST.

## Structure

The extracted code generally follows this structure:

```cpp
// Source file and line number
static void kernel_name_forward(
    wp::type arg1, 
    wp::array_t<type> arg2) 
{
    //---------
    // primal vars
    // Declaration of all local variables (SSA-like style)
    wp::float32 var_0;
    wp::int32 var_1;
    
    //---------
    // forward
    // Execution logic interspersed with comments from original Python source
    
    // def kernel(...): <Line Number>
    
    // x = a + b
    var_0 = wp::add(var_a, var_b);
    
    // Control flow (loops/conditionals)
    // May use explicit gotos and labels for loops
    start_for_0:;
        if (condition) goto end_for_0;
        ...
        goto start_for_0;
    end_for_0:;
}
```

## Key Characteristics

1. **Single Static Assignment (SSA) flavor**: Variables are often declared as `var_0`, `var_1` etc., though mutable variables exist.
2. **Warp Primitives**: Operations use `wp::add`, `wp::mul`, `wp::sin` etc., rather than standard C operators in many cases to support different types (vectors, matrices).
3. **Memory Access**: Array access is done via `wp::array_store` and `wp::load` (or `wp::address`).
4. **Control Flow**: Python loops (`for`, `while`) are often lowered to `goto` based state machines for generality, especially with custom iterators.
5. **Comments**: The original Python source lines are preserved as comments, aiding debugging and dataset alignment.

## Usability for LLM Training

This format is excellent for training LLMs on "Code Translation" or "Compiler Optimization" tasks because:
- It provides a ground-truth mapping between high-level Python and low-level C++ implementation.
- It exposes the "unrolled" or "lowered" logic (e.g., how a high-level `for` loop maps to pointer arithmetic and jumps).
- The embedded comments provide alignment data.
