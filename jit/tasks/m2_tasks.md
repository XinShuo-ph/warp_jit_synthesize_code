# Milestone 2 Tasks

## Task 1: Implement IR Extractor
- [x] Step 1.1: Create `code/extraction/ir_extractor.py`.
- [x] Step 1.2: Implement `get_kernel_ir(kernel)` function that uses `warp.codegen` to generate source.
- [x] Step 1.3: Handle CPU vs CUDA codegen (extract both or selectable).
- **Done when**: `get_kernel_ir` returns a string containing the C++/CUDA code for a given kernel.

## Task 2: Create Test Cases
- [x] Step 2.1: Create `code/examples/test_ir_extraction.py`.
- [x] Step 2.2: Define 5 diverse kernels (simple math, array access, loops, structs, atomic).
- [x] Step 2.3: Verify IR is extracted for all.
- **Done when**: 5+ test cases pass and print/save IR.

## Task 3: Document IR Format
- [x] Step 3.1: Analyze the extracted IR.
- [x] Step 3.2: Create `notes/ir_format.md`.
- **Done when**: `notes/ir_format.md` describes the structure of the generated code.
