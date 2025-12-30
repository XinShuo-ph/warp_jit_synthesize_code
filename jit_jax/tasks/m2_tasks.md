# Milestone 2 Tasks: IR Extraction Mechanism

## Task 1: IR Extraction Utility
- [x] Step 1.1: Create `code/extraction/ir_extractor.py`.
- [x] Step 1.2: Implement function `extract_ir(fn, *args)` that returns a dict with `jaxpr` and `hlo`.
- **Done when**: We can import `extract_ir` and get both IR formats from a python function.

## Task 2: Test Cases
- [x] Step 2.1: Create `code/extraction/test_extractor.py`.
- [x] Step 2.2: Add 5+ test cases (scalar math, vector math, control flow).
- [x] Step 2.3: Verify extraction works for all cases.
- **Done when**: `pytest` or manual run passes for all cases.

## Task 3: IR Documentation
- [x] Step 3.1: Analyze the extracted IR formats.
- [x] Step 3.2: Write `notes/ir_format.md` documenting the structure of HLO/StableHLO text.
- **Done when**: Documentation exists and explains the key components.
