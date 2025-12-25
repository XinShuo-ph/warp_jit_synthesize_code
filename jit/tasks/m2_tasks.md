# Milestone 2 Tasks: IR Extraction Mechanism

## Task 1: Prototype IR Extraction
- [ ] Step 1.1: Create `jit/code/extraction/ir_extractor.py` skeleton.
- [ ] Step 1.2: Implement a function `get_kernel_ir(kernel)` that accesses `kernel.adj`.
- [ ] Step 1.3: Experiment with triggering `kernel.adj.build()` or accessing pre-built blocks to retrieve C++ source.
- **Done when**: `get_kernel_ir` returns the C++ source string (or similar IR representation) for a simple kernel.

## Task 2: Refine Extraction & Formatting
- [ ] Step 2.1: Clean up the extracted IR (remove boilerplate if needed, focus on kernel body).
- [ ] Step 2.2: Ensure extraction works for both forward and backward (adjoint) passes if applicable.
- [ ] Step 2.3: Handle different output formats if necessary (e.g., returning a dictionary with forward/backward code).
- **Done when**: `ir_extractor.py` provides a clean API to get source code for a kernel.

## Task 3: Create Test Cases
- [ ] Step 3.1: Create `jit/code/extraction/test_ir_extractor.py`.
- [ ] Step 3.2: Add 5 diverse test kernels (arithmetic, loops, conditionals, array access, builtin calls).
- [ ] Step 3.3: Verify that extracted IR matches expected C++ patterns.
- **Done when**: 5+ test cases pass and verify Python -> C++ mapping.

## Task 4: Documentation
- [ ] Step 4.1: Analyze the structure of the extracted C++ code.
- [ ] Step 4.2: Write `jit/notes/ir_format.md`.
- **Done when**: `jit/notes/ir_format.md` describes the output format of the extractor.
