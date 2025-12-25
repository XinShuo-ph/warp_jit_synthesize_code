# Warp Basics: Kernel Compilation and IR

## Overview
Warp is a Python framework for high-performance simulation and graphics code.
Kernels are written in Python and JIT-compiled to C++/CUDA.

## Kernel Compilation Flow

1. **Kernel Definition** (@wp.kernel decorator)
   - Python function decorated with @wp.kernel
   - Type annotations are required for all parameters
   - Creates a Kernel object that belongs to a Module

2. **AST Analysis** (First reference)
   - Python AST is analyzed by codegen.py
   - Type checking and validation performed
   - Builds intermediate representation

3. **Code Generation** (On first wp.launch)
   - Module.load() is called
   - C++ code is generated from Python AST
   - Each operation becomes C++ function call (wp::add, wp::mul, etc.)
   - Automatic differentiation code is also generated (_backward function)

4. **Compilation** (C++ → Machine Code)
   - Generated C++ saved to cache: ~/.cache/warp/{version}/
   - Compiled to .o object file using C++ compiler
   - Loaded as shared library for execution

5. **Caching**
   - Hash computed from kernel code and options
   - Cache structure: wp_{module}_{hash}/
     - .cpp: Generated C++ code
     - .o: Compiled object file
     - .meta: Metadata (shared memory bytes, etc.)

## IR Format

The "IR" in Warp is C++ code with a structured format:

### Key Components:
- **Struct**: wp_args_{kernel_name}_{hash} holds kernel parameters
- **Forward function**: {kernel}_cpu_kernel_forward() contains main logic
- **Backward function**: {kernel}_cpu_kernel_backward() for autodiff
- **Entry points**: Exported C functions for Python to call

### Variable Naming:
- var_N: Primal (forward) variables
- adj_N: Adjoint (backward) variables
- Line numbers from Python source are preserved as comments

### Operations:
Python operators → wp:: namespace functions:
- a + b → wp::add(a, b)
- a * b → wp::mul(a, b)  
- wp.sin(x) → wp::sin(x)
- array[i] → wp::address() + wp::load()
