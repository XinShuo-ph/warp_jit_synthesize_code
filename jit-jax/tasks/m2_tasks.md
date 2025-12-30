# Milestone 2 Tasks: IR Extraction Mechanism

## Task 1: Create IR Extractor Module
- [x] Step 1.1: Create ir_extractor.py with extract_jaxpr() function
- [x] Step 1.2: Add extract_stablehlo() function
- [x] Step 1.3: Add extract_compiled_hlo() function
- [x] Step 1.4: Add unified extract_ir() function with format option
- **Done when**: All extraction functions work on simple test function

## Task 2: Create Test Cases
- [x] Step 2.1: Test case 1 - simple arithmetic
- [x] Step 2.2: Test case 2 - matrix operations
- [x] Step 2.3: Test case 3 - gradient computation
- [x] Step 2.4: Test case 4 - control flow (jax.lax.cond)
- [x] Step 2.5: Test case 5 - scan/loop operations
- **Done when**: All 5 test cases produce valid Python→IR pairs

## Task 3: Document IR Format
- [x] Step 3.1: Create notes/ir_format.md
- [x] Step 3.2: Document StableHLO operation types
- [x] Step 3.3: Document type notation
- **Done when**: notes/ir_format.md exists (max 30 lines)
