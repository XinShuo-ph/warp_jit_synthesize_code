# Milestone 4 Tasks: Synthesis Pipeline

## Task 1: Code Generator
- [x] Step 1.1: Create `code/synthesis/generator.py`.
- [x] Step 1.2: Implement `generate_random_function()` that produces a valid Python string containing a JAX function.
- [x] Step 1.3: Ensure generated code uses `jax.numpy` operations.
- **Done when**: We can generate 10 unique valid Python functions.

## Task 2: Pipeline
- [x] Step 2.1: Create `code/synthesis/pipeline.py`.
- [x] Step 2.2: Implement `process_generated_code(code_str)` which:
    - Execs the code to get the function.
    - Generates random inputs.
    - Runs `extract_ir`.
    - Returns the pair (code, ir).
- **Done when**: We can go from generated string to IR.

## Task 3: Sample Generation
- [x] Step 3.1: Run the pipeline to generate 100 samples.
- [x] Step 3.2: Save them to `jit_jax/data/samples/`.
- [x] Step 3.3: Verify a few samples manually.
- **Done when**: 100+ files exist in data directory.
