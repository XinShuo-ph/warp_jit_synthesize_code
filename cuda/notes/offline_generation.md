# Offline CUDA IR Generation

## Overview
We successfully implemented a pipeline to generate CUDA intermediate representation (CUDA C++ code) without requiring a physical GPU or NVIDIA drivers. This enables dataset generation on CPU-only nodes (e.g., CI/CD environments, standard cloud instances).

## Technique

### The Problem
Standard `warp` usage requires a CUDA driver to be present to initialize the CUDA context. Calling `module.load("cuda")` triggers both code generation and compilation (via `nvcc`), failing if drivers are missing.

### The Solution
We bypassed the runtime driver checks by directly using Warp's internal `ModuleBuilder` class.

1. **ModuleBuilder**: Located in `warp.context` (or `warp._src.context`), this class handles the translation of Warp Modules/Kernels into C++/CUDA source code.
2. **Options Injection**: We must manually supply `output_arch` in the builder options to satisfy internal checks (e.g., set to 86 for Ampere).
3. **Manual Hashing**: We call `module.hash_module()` explicitly since we aren't loading the module via the runtime.

### Implementation Details
The core logic in `pipeline_offline.py`:

```python
from warp.context import ModuleBuilder

# ... setup module ...
module.hash_module()

builder_options = module.options.copy()
builder_options["output_arch"] = 86  # Mock architecture

builder = ModuleBuilder(module, builder_options, hasher=...)
cuda_source = builder.codegen("cuda")
```

This generates the full `.cu` source code string, which we then parse using our existing regex-based IR extractor.

## Validation
We verified that the generated code contains:
- Correct CUDA kernel signatures (`__global__ void ...`).
- Device-specific memory access patterns (`blockDim`, `threadIdx`).
- Expected Warp built-in calls (`wp::load`, `wp::mul`).

## Usage
To generate offline samples:
```bash
python3 cuda/code/synthesis/pipeline_offline.py --count 100 --output ../../data/offline_dataset
```
