# GPU Analysis

## Current CUDA Support
- ir_extractor.py has device param: **No** (currently hardcoded to use whatever device compiled the kernel)
- Tested with device="cuda": **No GPU available** (environment has no CUDA driver)
- The extractor already captures `.cu` files if they exist (line 176-177 in ir_extractor.py)

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| File extension | `.cpp` | `.cu` |
| Block dim define | `#define WP_TILE_BLOCK_DIM 1` | `#define WP_TILE_BLOCK_DIM <block_size>` |
| Kernel naming | `*_cpu_kernel_forward` | `*_cuda_kernel_forward` |
| Entry point | `*_cpu_forward` | `*_cuda_forward` |
| Thread indexing | `wp::tid(task_index, dim)` | Uses CUDA `threadIdx`, `blockIdx` |
| Execution model | Serial loop over `task_index` | CUDA grid/block launch |
| Memory | Stack variables | Shared memory (`__shared__`) possible |
| Meta tracking | `*_cuda_kernel_*_smem_bytes: 0` | `*_cuda_kernel_*_smem_bytes: <size>` |

## Key Observations from Current Code

1. **Metadata already tracks CUDA**: The `.meta` JSON file includes fields like:
   ```json
   {"map_1_2ea93cd6_cuda_kernel_forward_smem_bytes": 0}
   ```
   Even when running CPU-only, warp generates metadata anticipating CUDA.

2. **Dual code generation**: Warp generates both `.cpp` (CPU) and `.cu` (CUDA) files when CUDA is available. The `_find_module_files()` method already handles both.

3. **Naming convention**: 
   - CPU: `kernel_<hash>_cpu_kernel_forward()`
   - GPU: `kernel_<hash>_cuda_kernel_forward()`

4. **Block execution**:
   - CPU: Sequential for-loop over all threads
   - GPU: Parallel execution across blocks/threads

## Changes Needed for GPU

1. **Add `device` parameter to pipeline**:
   ```python
   # In pipeline.py and batch_generator.py
   def generate_sample(seed, device="cpu"):
       # Launch with device parameter
       wp.launch(kernel, dim=n, inputs=[...], device=device)
   ```

2. **Modify ir_extractor to return device-specific IR**:
   ```python
   def extract_ir(self, kernel, device="cpu"):
       # Return .cu for cuda, .cpp for cpu
       if device == "cuda":
           return module_files.get('cuda')
       return module_files.get('cpp')
   ```

3. **Update KernelIR dataclass**:
   ```python
   @dataclass
   class KernelIR:
       python_source: str
       cpp_code: str      # Could rename to 'ir_code'
       cuda_code: str     # New field for CUDA IR
       device: str        # 'cpu' or 'cuda'
       ...
   ```

4. **Handle GPU-specific array creation**:
   ```python
   # Arrays need to be on correct device
   a = wp.array([1.0, 2.0], dtype=float, device=device)
   ```

## New GPU-Specific Patterns to Add

- [ ] **Tile operations**: `wp.tile_load`, `wp.tile_store` for shared memory
- [ ] **Block synchronization**: `wp.synchronize_block()` 
- [ ] **Atomic operations on GPU**: Different performance characteristics
- [ ] **Cooperative groups**: For advanced GPU patterns
- [ ] **Tensor cores**: WMMA operations for matrix multiply
- [ ] **Warp-level primitives**: `wp.shfl`, warp shuffle operations

## Testing Recommendations

1. **Test on GPU-enabled machine**:
   ```bash
   # Check CUDA availability
   python3 -c "import warp as wp; wp.init(); print(wp.is_cuda_available())"
   ```

2. **Generate paired CPU/GPU samples**:
   ```python
   # Generate same kernel on both devices
   for device in ["cpu", "cuda"]:
       ir = extract_kernel_ir(kernel, device=device)
       save_sample(ir, suffix=f"_{device}")
   ```

3. **Validate GPU-specific features**:
   - Shared memory usage
   - Block dimensions
   - Coalesced memory access patterns

## Conclusion

The existing infrastructure is largely GPU-ready:
- Extractor already looks for `.cu` files
- Meta file tracks CUDA shared memory
- Naming conventions distinguish CPU/GPU

Main gaps:
- No `device` parameter exposed in pipeline
- Need CUDA-enabled hardware to test and generate `.cu` samples
- Dataset currently CPU-only but format could easily include both
