# CUDA Backend Adaptation Notes

## Overview
The codebase has been adapted to support both CPU and CUDA backends for the JIT synthesis pipeline.

## Key Changes

### 1. Pipeline (`pipeline.py`)
- Added `--device` argument to CLI and `SynthesisPipeline` class.
- `compile_kernel_from_source` now accepts a `device` parameter and loads the module on that device: `wp_module.load(device)`.
- `generate_pair` logic looks for `.cu` files instead of `.cpp` files when `device="cuda"`.
- IR extraction is device-aware.

### 2. IR Extraction (`ir_extractor.py`)
- `extract_ir` now locates the correct source file based on device:
  - CPU: `~/.cache/warp/.../module_id.cpp`
  - CUDA: `~/.cache/warp/.../module_id.cu`
- `extract_kernel_functions` regex patterns updated to include `_{device}_kernel_forward` naming convention (assumed based on standard Warp patterns).

### 3. Testing (`test_basic_kernels.py`)
- Updated to accept `--device` argument.
- Kernels now launch on the specified device.
- Arrays are allocated on the specified device.

## Verification
Since no GPU was available during development, the logic assumes standard Warp behavior for CUDA code generation. 
If compilation fails or regexes don't match on actual hardware:
1. Check the generated `.cu` file content in `~/.cache/warp/...`.
2. Update `ir_extractor.py` regexes to match the actual function signatures.
