# Milestone 4 Tasks: Offline CUDA IR Generation

## Task 1: Investigate Warp Codegen
- [ ] Inspect `warp` package internals (via script) to find `CodeGen` class or similar.
- [ ] Attempt to instantiate `CodeGen` or call `build_cuda` manually.
- **Done when**: A method to generate CUDA source string from a `wp.Kernel` is identified.

## Task 2: Implement Offline Pipeline
- [ ] Create `cuda/code/synthesis/pipeline_offline.py`.
- [ ] Implement `generate_cuda_source(kernel)` function that bypasses driver check.
- [ ] Integrate into synthesis loop.
- **Done when**: `python pipeline_offline.py` produces `.cu` content (even if not compiled).

## Task 3: Verify & Document
- [ ] Generate 3-5 samples.
- [ ] Verify content looks like valid CUDA C++.
- [ ] Write `notes/offline_generation.md`.
- **Done when**: Offline generation is proven to work.
