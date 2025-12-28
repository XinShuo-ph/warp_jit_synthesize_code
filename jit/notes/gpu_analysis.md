# GPU Analysis

## Current CUDA Support
- **ir_extractor.py has device param**: No
- **Tested with device="cuda"**: No GPU available in environment (CPU-only mode)

## Key Finding: IR is Device-Independent

The Warp IR extracted by `ir_extractor.py` is **device-independent**. The `adj.blocks[*].body_forward` contains the same SSA-style operations regardless of target device. The device-specific differences only appear in:
1. The kernel template wrapper (CUDA vs CPU)
2. Memory management / launch configuration
3. Thread indexing pattern

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| **File Extension** | `.cpp` | `.cu` |
| **Module Header** | `cpu_module_header` | `cuda_module_header` |
| **Kernel Declaration** | `void {name}_cpu_kernel_forward(...)` | `extern "C" __global__ void {name}_cuda_kernel_forward(...)` |
| **Thread Loop** | Single task index passed as arg | Grid-stride loop: `for (_idx = blockDim.x * blockIdx.x + threadIdx.x; ...)` |
| **Args Passing** | Via struct pointer (`wp_args_{name}*`) | Direct parameters in function signature |
| **Return in Kernel** | `return;` | `continue;` (due to grid-stride loop) |
| **Shared Memory** | Not used | `wp::tile_shared_storage_t tile_mem;` |
| **Compiler** | LLVM (via `wp_compile_cpp`) | NVRTC (via `wp_cuda_compile_program`) |

## IR Body (Extracted Content) - Same for Both

The IR extracted by `ir_extractor.py` comes from `adj.blocks[*].body_forward`, which is identical for CPU and GPU:

```cpp
// Block 0
var_0 = builtin_tid1d();          // Thread index
var_1 = wp::address(var_x, var_0); // Memory address calculation
var_2 = wp::load(var_1);           // Load from memory
var_3 = wp::add(var_2, var_4);     // Arithmetic operation
wp::array_store(var_x, var_0, var_3); // Store to memory
```

These `wp::` namespace operations compile to equivalent implementations on both CPU and GPU.

## Changes Needed for GPU Output

### 1. Add Device Parameter (Optional Enhancement)
The current `ir_extractor.py` extracts the abstract IR which is device-agnostic. To extract the **full generated source code** (not just IR), modifications would be needed:

```python
def extract_ir(kernel_func, device="cpu"):
    """
    Args:
        device: "cpu" or "cuda" - for full source output only
    """
    # ... build the kernel ...
    
    # To get full device-specific source:
    # source = builder.codegen(device)  # This generates .cpp or .cu content
```

### 2. Capture Full Generated Code
To capture the complete `.cpp` or `.cu` source (not just IR), the extractor would need to:
- Call `builder.codegen("cpu")` or `builder.codegen("cuda")`
- Return the full template-wrapped code

### 3. Add Device Metadata to Samples
Current samples don't specify device. Add:
```json
{
  "device": "cpu",  // or "cuda"
  "ir_format": "body_forward",
  "full_source": "..."  // optional: complete .cpp/.cu code
}
```

## New GPU-Specific Patterns to Add

For a GPU-focused dataset, consider generating samples with:

- [ ] **Tile operations**: `wp.tile_load`, `wp.tile_store`, `wp.tile_reduce`
- [ ] **Shared memory usage**: Patterns using `tile_shared_storage_t`
- [ ] **Warp-level primitives**: `__syncthreads()`, warp shuffles
- [ ] **Grid-stride loop patterns**: Multi-element-per-thread processing
- [ ] **Atomic operations**: `wp.atomic_add`, `wp.atomic_min`, etc.
- [ ] **Block/grid dimension awareness**: `blockDim.x`, `gridDim.x` in patterns
- [ ] **Memory coalescing patterns**: Access patterns optimized for GPU memory

## Warp's Code Generation Pipeline

```
Python Kernel (@wp.kernel)
        ↓
    Adjoint Pass (adj.blocks[*].body_forward) ← THIS IS WHAT WE EXTRACT
        ↓
    codegen(device)
        ↓
  ┌─────┴─────┐
  ↓           ↓
CPU (.cpp)  CUDA (.cu)
  ↓           ↓
LLVM        NVRTC
  ↓           ↓
.obj        .ptx/.cubin
```

## Conclusion

The current IR extraction approach is **already GPU-compatible** because:
1. The extracted IR (`body_forward`) is device-agnostic
2. The `wp::` operations map to both CPU and CUDA implementations
3. An LLM trained on this data would learn the core computation pattern

To specifically train for CUDA code generation, consider:
1. Extracting full `.cu` source via `builder.codegen("cuda")`
2. Adding CUDA-specific patterns (tiles, atomics, sync primitives)
3. Including launch configuration in samples (block size, grid size)
