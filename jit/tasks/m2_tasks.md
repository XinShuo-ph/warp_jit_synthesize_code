# Milestone 2 Tasks

## Task 1: Create Extraction Utility
- [ ] Step 1.1: Create `jit/code/extraction/ir_extractor.py`.
- [ ] Step 1.2: Implement function `extract_ir(kernel_func) -> str`.
- [ ] Step 1.3: Handle module and builder initialization cleanly.
- **Done when**: A clean python module exists that can extract IR from any provided kernel.

## Task 2: Create Test Cases
- [ ] Step 2.1: Create `jit/code/extraction/test_extractor.py`.
- [ ] Step 2.2: Add tests for simple math kernel.
- [ ] Step 2.3: Add tests for control flow (if/else, loops).
- [ ] Step 2.4: Add tests for structs/arrays.
- **Done when**: 5+ test cases pass and output valid Python-IR pairs.

## Task 3: Document IR Format
- [ ] Step 3.1: Analyze the extracted IR from tests.
- [ ] Step 3.2: Write `jit/notes/ir_format.md` detailing the structure.
- **Done when**: Documentation exists describing the fields and format of extracted data.
