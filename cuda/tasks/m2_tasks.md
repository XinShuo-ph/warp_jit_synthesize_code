# Milestone 2 Tasks: CUDA Adaptation

## Task 1: Pipeline Adaptation
- [ ] Modify `pipeline.py`: Add `device` parameter to `compile_kernel_from_source`.
- [ ] Modify `pipeline.py`: Update `extract_kernel_ir` regex to support `cuda` patterns.
- [ ] Modify `pipeline.py`: Add CLI argument `--device`.
- **Done when**: `pipeline.py` can theoretically target CUDA and extract IR (verified via code inspection or mock).

## Task 2: IR Extractor Update
- [ ] Check `code/extraction/ir_extractor.py` for similar CPU-hardcoded patterns.
- [ ] Update if necessary.
- **Done when**: All extraction tools support CUDA.

## Task 3: Kernel Iteration (Dry Run)
- [ ] Attempt to generate/compile kernels with `device="cuda"`.
- [ ] **NOTE**: This will likely fail without a GPU driver.
- [ ] Create a "dry-run" or "force-generate" mode if possible, or provide instructions for the user.
- **Done when**: We have a plan for the user to run this.

## Task 4: User Verification Script
- [ ] Create `verify_cuda.py` that runs the pipeline on CUDA and checks output.
- **Done when**: Script exists and is documented.
