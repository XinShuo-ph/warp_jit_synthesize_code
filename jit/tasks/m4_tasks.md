# Milestone 4 Tasks: Synthesis Pipeline

## Task 1: Kernel Generator
- [x] Step 1.1: Create `jit/code/synthesis/generator.py`.
- [x] Step 1.2: Implement a class `KernelGenerator` that produces valid Python source strings for Warp kernels.
- [x] Step 1.3: Support randomization: variable number of inputs, different math operations (`+`, `*`, `sin`, `exp`), loop depths.
- **Done when**: `generator.generate_kernel_source()` returns a valid, compilable Python string.

## Task 2: Pipeline Integration
- [x] Step 2.1: Create `jit/code/synthesis/pipeline.py`.
- [x] Step 2.2: Implement `synthesize_pair()`:
    1. Generate Python source.
    2. Dynamic loading (using `exec` or writing to temp file and importing).
    3. Extract IR using `ir_extractor.py`.
    4. Save (Python Source, IR) pair to JSONL.
- **Done when**: `pipeline.py` runs and produces a JSONL file with valid pairs.

## Task 3: Batch Generation
- [x] Step 3.1: Run the pipeline to generate 100 samples.
- [x] Step 3.2: Verify diversity (check unique hashes or source strings).
- [x] Step 3.3: Ensure compilation doesn't crash on invalid generated code (robustness).
- **Done when**: 100 varied pairs are saved in `jit/data/samples/`.

## Task 4: FEM Kernel Synthesis (Optional/Bonus)
- [ ] Step 4.1: Generate kernels that look like FEM integrands (using `fem` decorators).
- **Done when**: At least 10 FEM-style kernels are generated.
