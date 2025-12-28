# Milestone 3 Tasks

## Context
The current generators create device-agnostic Python kernels that work with both CPU and CUDA backends.
Warp's codegen automatically translates them to device-specific code.

However, to fully demonstrate CUDA capabilities, we should add CUDA-specific patterns:
- Explicit thread/block indexing
- Shared memory usage
- More sophisticated atomic operations
- Thread synchronization

## Task 1: Arithmetic Kernels - CUDA Enhancement
- [x] Verify current arithmetic kernels work with CUDA backend
- [x] Create test example showcasing CUDA thread indexing
- [ ] Document CUDA-specific IR patterns
- **Done when**: Test runs successfully and generates CUDA IR with thread indexing

## Task 2: Math Kernels - CUDA Enhancement
- [ ] Verify current math kernels work with CUDA backend
- [ ] Create test example with multiple math operations
- [ ] Document CUDA math library usage in IR
- **Done when**: Test runs successfully and math operations visible in CUDA IR

## Task 3: Loop Kernels - CUDA Parallel Patterns
- [ ] Add explicit parallel loop pattern using wp.tid()
- [ ] Test reduction patterns on CUDA
- [ ] Document CUDA loop unrolling in IR
- **Done when**: Loop kernels generate efficient CUDA code

## Task 4: Conditional Kernels - Branch Divergence
- [ ] Verify conditional patterns work with CUDA
- [ ] Test nested conditionals
- [ ] Document branch divergence handling in CUDA IR
- **Done when**: Conditional kernels compile and document divergence

## Task 5: Vector Kernels - CUDA Vector Types
- [ ] Verify vector operations (vec2, vec3, vec4) work with CUDA
- [ ] Test all vector operations (dot, cross, normalize)
- [ ] Document CUDA vector type handling
- **Done when**: All vector operations work on CUDA backend

## Task 6: Matrix Kernels - CUDA Matrix Operations
- [ ] Verify matrix operations work with CUDA
- [ ] Test matrix-vector and matrix-matrix multiply
- [ ] Document memory access patterns in CUDA IR
- **Done when**: Matrix operations generate efficient CUDA code

## Task 7: Atomic Kernels - CUDA Atomics
- [ ] Current atomic kernels already use wp.atomic_* operations
- [ ] Verify CUDA atomic operations in IR
- [ ] Add test for all atomic types (add, min, max)
- [ ] Document CUDA atomic instruction usage
- **Done when**: Atomic operations verified in CUDA IR

## Notes
- Most generators already work with CUDA out-of-box
- Focus is on validation and documentation
- CUDA-specific optimizations are handled by Warp compiler
- Main value: documenting what CUDA patterns look like
