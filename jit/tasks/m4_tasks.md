# Milestone 4: Synthesis Pipeline

## Task 1: Random JAX Code Generator
- [x] Create `jit/code/synthesis/generator.py`.
- [x] Implement `generate_random_jax_fn()` which returns a string of python code.
- [x] Must support basic math, array ops, and maybe control flow (lax.cond/scan).
- **Done when**: We can generate valid, executable python strings.

## Task 2: Execution & Extraction Pipeline
- [x] Create `jit/code/synthesis/pipeline.py`.
- [x] Function to take code string -> compile -> extract IR -> save.
- [x] Handle compilation errors (filtering invalid generated code).
- **Done when**: End-to-end flow works for a generated sample.

## Task 3: Batch Generation
- [x] Generate 100 samples.
- [x] Save to `jit/data/samples/`.
- **Done when**: 100 pairs exist.
