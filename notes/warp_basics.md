# Warp Basics: Kernel Compilation & IR

## Overview
Warp compiles Python kernels to C++/CUDA through a JIT compilation pipeline. The "IR" (intermediate representation) is the generated C++/CUDA code.

## Kernel Compilation Flow

1. **Python Definition**: User writes kernel with `@wp.kernel` decorator
2. **AST Parsing**: Warp parses the Python function using `ast` module
3. **Code Generation**: `warp._src.codegen.py` generates C++ code
4. **Compilation**: Generated C++ is compiled to native binary
5. **Caching**: Binary and source cached in `~/.cache/warp/VERSION/`

## Key Components

### Module System
- Each Python file becomes a warp Module
- Module contains one or more kernels
- Module hash computed from source code
- Cached in `wp___<module_name>___<hash>/` directory

### Generated Files
- `.cpp` - C++ source (the IR we extract)
- `.o` - Compiled object file
- `.meta` - JSON metadata (shared memory, etc.)
- `.cu` - CUDA source (if CUDA available)

## IR Structure

Generated C++ contains:
- Forward kernel function (main computation)
- Backward kernel function (for autodiff)
- Argument struct definition
- Entry point wrappers

Example mapping:
```python
@wp.kernel
def add(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = a[tid] + 1.0
```

Generates (~900 lines of C++):
- Argument struct: `wp_args_add_<hash>`
- Forward: `add_<hash>_cpu_kernel_forward`
- Backward: `add_<hash>_cpu_kernel_backward`
- Entry points: `add_<hash>_cpu_forward/backward`

## IR Extraction Method

Located in `code/extraction/ir_extractor.py`:
1. Compile kernel (by launching it)
2. Find module cache directory
3. Read `.cpp` file from cache
4. Pair with original Python source

## Key Insights

- IR is deterministic - same Python → same C++
- Each kernel gets unique hash based on signature
- Backward pass auto-generated for autodiff
- Type annotations critical - all inferred from Python
- Variables in loops need explicit typing: `x = float(0.0)`
