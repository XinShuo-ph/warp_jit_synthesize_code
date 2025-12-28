# Milestone 4 Tasks

## Context
Forward and backward pass support for CUDA kernels. The ir_extractor already has
`include_backward` parameter that generates adjoint (backward) kernels.

## Task 1: Forward Pass CUDA Support
- [x] Verify forward pass works for all kernel types
- [x] Test with include_backward=False
- [x] Document forward kernel naming convention
- **Done when**: All tests pass with forward-only generation

## Task 2: Backward Pass CUDA Support
- [ ] Test backward/adjoint kernel generation with include_backward=True
- [ ] Verify gradient computation kernels compile for CUDA
- [ ] Create examples showing forward+backward pairs
- [ ] Document backward kernel IR structure
- **Done when**: Backward kernels generate successfully for CUDA

## Notes
- Forward pass already validated in M3 tests
- Backward pass support is built into Warp's autodiff system
- Just need to verify it works with CUDA backend
