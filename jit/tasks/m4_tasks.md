# Milestone 4 Tasks

## Task 1: Kernel Generator
- [ ] Step 1.1: Create `jit/code/synthesis/generator.py`.
- [ ] Step 1.2: Implement `generate_random_kernel_code() -> str`.
- [ ] Step 1.3: Support basic math operations and loop constructs.
- **Done when**: Can generate valid Python strings representing Warp kernels.

## Task 2: Synthesis Pipeline
- [ ] Step 2.1: Create `jit/code/synthesis/pipeline.py`.
- [ ] Step 2.2: Integrate `generator` and `ir_extractor`.
- [ ] Step 2.3: Implement `generate_pair()`:
    1. Generate Python code.
    2. Dynamically load/exec it.
    3. Extract IR.
    4. Save to disk.
- **Done when**: Running the pipeline produces JSON files with Python/IR pairs.

## Task 3: Data Generation
- [ ] Step 3.1: Generate 100 sample pairs.
- [ ] Step 3.2: Verify diversity and validity (manual spot check).
- **Done when**: `jit/data/samples/` contains 100+ valid pairs.
