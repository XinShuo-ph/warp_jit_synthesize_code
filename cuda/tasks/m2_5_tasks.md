# Milestone 2.5 Tasks

## Task 1: Update Pipeline for Headless Mode
- [x] Step 1.1: Modify `pipeline.py` to detect if `module.load("cuda")` is possible.
- [x] Step 1.2: Implement fallback using `warp._src.context.ModuleBuilder(module, options, hasher).codegen("cuda")`.
- [x] Step 1.3: Update `ir_extractor.py` usage in `pipeline.py` to handle source string directly.
- **Done when**: `pipeline.py` can produce a `SynthesisPair` with CUDA IR even when running on a CPU machine.

## Task 2: Verify Headless Generation
- [x] Step 2.1: Create `cuda/code/backend/test_headless.py`.
- [x] Step 2.2: Run the test and verify output JSON contains CUDA code.
- **Done when**: `test_headless.py` passes and generates valid JSON files with CUDA IR.
