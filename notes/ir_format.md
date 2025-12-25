# Warp IR Format Documentation

## Structure Overview
Generated C++ code has three main sections:
1. Header/Definitions
2. Function implementations (forward + backward)
3. Kernel implementations (forward + backward)

## 1. Header Section
- Standard includes: `builtin.h`
- Type casting macros: `float(x)`, `int(x)`, `adj_float()`, `adj_int()`
- Built-in function macros: `builtin_tid1d()`, `builtin_tid2d()`, etc.

## 2. SSA Variable Naming
All intermediate values use Static Single Assignment (SSA):
- `var_0`, `var_1`, ...: Forward pass variables
- `adj_0`, `adj_1`, ...: Backward pass (adjoint) variables
- Constants: `const wp::float32 var_2 = 0.01`

## 3. Function Signature Patterns
```cpp
// Forward pass
static ReturnType func_name_suffix(ArgType1 var_arg1, ArgType2 var_arg2)

// Backward pass (adjoint)
static void adj_func_name_suffix(
    ArgType1 var_arg1,        // primal inputs
    ArgType2 var_arg2,
    ArgType1& adj_arg1,       // gradient inputs/outputs
    ArgType2& adj_arg2,
    ReturnType& adj_ret)      // gradient of return value
```

## 4. Kernel Structure
Each kernel generates:
- Argument struct: `wp_args_{kernel_name}_{hash}`
- Forward kernel: `{kernel_name}_{hash}_cpu_kernel_forward`
- Backward kernel: `{kernel_name}_{hash}_cpu_kernel_backward`
- Entry points: `{kernel_name}_{hash}_cpu_forward` (extern "C")

## 5. Control Flow Translation
- If/else: Direct C++ translation with goto labels for adjoint
- Loops: `wp::range()` iterator, reversed in adjoint pass
- Early returns: Use goto labels (`label0:`, `label1:`)
