# Milestone 2 Tasks

## Task 1: Create Backend Infrastructure
- [x] Step 1.1: Set up `cuda/code/backend` by copying `cuda/code/base`.
- [x] Step 1.2: Refactor `pipeline.py` to accept a `device` argument (cpu/cuda).
- [x] Step 1.3: Update `ir_extractor.py` to handle CUDA IR extraction (file extensions, function names).
- **Done when**: The pipeline code supports the `--device` flag and attempts to use the CUDA backend.

## Task 2: Validation Script
- [x] Step 2.1: Create a test script `cuda/code/backend/test_cuda.py` that the user can run.
- [x] Step 2.2: Document how to run it.
- **Done when**: A clear entry point for CUDA testing exists.
