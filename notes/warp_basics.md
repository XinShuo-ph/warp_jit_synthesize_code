# Warp Compilation Flow and IR Location

## Overview
Warp compiles Python kernel functions into C++ code, which is then compiled to native machine code (CPU) or PTX (GPU).

## Compilation Flow

### 1. Kernel Definition
- User decorates a function with `@wp.kernel`
- Creates a `Kernel` object (defined in `_src/context.py`)
- Kernel is registered with its parent `Module`

### 2. First Launch
- When `wp.launch()` is called for the first time, compilation is triggered
- Module.load() is called on the target device
- Module.compile() generates source code via codegen

### 3. Code Generation (IR Creation)
- `codegen_kernel()` function in `_src/codegen.py` (line ~4290)
- Parses Python AST of the kernel function
- Creates intermediate `Var` objects representing operations (line ~673)
- Generates C++ source code with explicit variable assignments
- Each Python operation becomes a C++ function call (e.g., `wp::mul()`, `wp::add()`)

### 4. Output Structure
Generated C++ code includes:
- Argument struct: `wp_args_<kernel_name>_<hash>`
- Forward function: `<kernel_name>_<hash>_cpu_kernel_forward()`
- Backward function (for autodiff): `<kernel_name>_<hash>_cpu_kernel_backward()`
- Each variable is explicitly declared (e.g., `wp::int32 var_0`, `wp::float32 var_3`)

### 5. Cache and Compilation
- Generated C++ saved to: `~/.cache/warp/<version>/wp_<module>_<hash>/`
- Files created:
  - `.cpp`: Generated C++ source code (the IR we want)
  - `.o`: Compiled object file
  - `.meta`: Metadata about the module
- C++ is compiled with native compiler (gcc/clang for CPU, nvcc for CUDA)

## Key Files for M2

### Core Implementation
- `warp/_src/context.py`: Module and Kernel classes
- `warp/_src/codegen.py`: Code generation (4391 lines total)
  - `Var` class (line 673): Intermediate representation variables
  - `codegen_kernel()` (line 4290): Main kernel codegen function
  - `codegen_module()` (line 4372): Module-level codegen

### Accessing Generated IR
1. After kernel launch, check cache directory at `wp.config.kernel_cache_dir`
2. Find module directory: `wp_<module_name>_<hash>/`
3. Read `.cpp` file for full IR

## Example IR Structure
```cpp
// Variable declarations
wp::int32 var_0;           // Thread ID
wp::float32* var_1;        // Array pointer
wp::float32 var_3;         // Intermediate result

// Operations
var_0 = builtin_tid1d();                    // Python: tid = wp.tid()
var_1 = wp::address(var_a, var_0);         // Python: a[tid]
var_4 = wp::load(var_1);
var_3 = wp::mul(var_4, var_2);             // Python: a[tid] * 2.0
```

## For M2: Extraction Strategy
1. Compile kernel to trigger IR generation
2. Locate cache file using module name/hash
3. Parse `.cpp` file to extract kernel function
4. Pair with original Python source code
