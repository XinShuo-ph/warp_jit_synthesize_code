# Milestone 4 Tasks

## Task 1: Kernel Generator Implementation
- [ ] Step 1.1: Create `code/synthesis/generator.py`.
- [ ] Step 1.2: Implement `generate_kernel_code()` that produces a string containing a valid `@wp.kernel` function.
    - Support random variations: arithmetic ops, array access, loops, conditionals.
- [ ] Step 1.3: Create a test script to verify generated code is valid Python and can be compiled by Warp.
- **Done when**: `generator.py` can produce 10+ distinct, compilable kernel strings.

## Task 2: Synthesis Pipeline
- [ ] Step 2.1: Create `code/synthesis/pipeline.py`.
- [ ] Step 2.2: Implement `generate_pair()`:
    1. Generate kernel code.
    2. Load kernel into current process (dynamic execution).
    3. Extract IR using `ir_extractor.py`.
    4. Return (python_code, ir_code).
- [ ] Step 2.3: Implement saving mechanism (e.g., JSONL or separate files).
- **Done when**: `pipeline.py` runs and saves a single Python-IR pair successfully.

## Task 3: Dataset Generation
- [ ] Step 3.1: Run pipeline to generate 100 samples in `data/samples/`.
- [ ] Step 3.2: Verify a random subset of samples manually (check Python syntax and IR presence).
- **Done when**: 100+ valid pairs exist in `data/samples/`.
