# Baseline Analysis

## Overview
The baseline is a data synthesis pipeline that generates pairs of (Python Source, C++ IR) for training LLMs. It uses NVIDIA Warp (`warp-lang`) to compile Python kernels and extract the intermediate C++ representation.

## Core Components
1.  **Generator (`cuda/code/base/synthesis/generator.py`)**:
    -   Classes: `KernelGenerator`, `KernelSpec`
    -   Generates random kernels of various types:
        -   `arithmetic`: Simple element-wise ops
        -   `conditional`: if/else
        -   `loop`: for loops
        -   `math`: sin/cos etc.
        -   `vector`: vec3 ops
        -   `atomic`: atomic_add
        -   `nested`: nested loops
        -   `multi_cond`: multiple if/else
        -   `combined`: complex logic
        -   `scalar_param`: mixed array/scalar args

2.  **Pipeline (`cuda/code/base/synthesis/pipeline.py`)**:
    -   Generates Python source using `KernelGenerator`.
    -   Compiles using `warp` (JIT).
    -   Extracts C++ IR from the `warp` cache.
    -   Currently hardcoded to `wp_module.load("cpu")`.
    -   Extracts `_cpu_kernel_forward` and `backward` symbols.

3.  **Extractor (`cuda/code/base/extraction/ir_extractor.py`)**:
    -   Helper to read `.cpp` files from `~/.cache/warp/...`.
    -   Regex parsing for function bodies.

## CUDA Porting Requirements
To adapt for CUDA:
1.  **Backend Selection**: Change `wp_module.load("cpu")` to `wp_module.load("cuda")`.
2.  **IR Extraction**:
    -   Warp generates `.cu` files for CUDA, not `.cpp` (usually).
    -   Function names will likely differ (e.g., `_cuda_kernel_forward` or similar).
    -   The path in cache might be different.
3.  **Environment**: Requires `warp` to detect a CUDA device or force CUDA codegen.

## Next Steps
-   Reproduce CPU execution to confirm baseline works.
-   Modify `pipeline.py` to support backend switching.
-   Investigate Warp's CUDA output format.
    -   Note: Since no GPU is available, actual execution of `wp.load("cuda")` might fail. The goal is to provide code that *will* work on a GPU machine. I can verify the logic changes (file paths, regex patterns) by reading Warp source or documentation if available, or making reasonable assumptions and providing a test script.
