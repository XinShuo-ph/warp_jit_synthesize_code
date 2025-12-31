# Milestone 2 Tasks: IR Extraction Mechanism

## Task 1: Enhance IR extractor
- [x] Step 1.1: Add batch extraction support
- [x] Step 1.2: Add JSON/dict output format for easy serialization
- [x] Step 1.3: Handle edge cases (vmap, grad, etc.)
- **Done when**: Extractor handles varied JAX transformations ✓

## Task 2: Create 5+ test cases
- [x] Step 2.1: Basic arithmetic functions
- [x] Step 2.2: Matrix operations (matmul, transpose, reshape)
- [x] Step 2.3: Activation functions (relu, softmax, gelu)
- [x] Step 2.4: Reduction operations (sum, mean, max)
- [x] Step 2.5: JAX transformations (vmap, grad)
- [x] Step 2.6: Control flow (cond, scan)
- **Done when**: All test cases generate valid Python→HLO pairs ✓ (8/8 pass)

## Task 3: Create test suite
- [x] Step 3.1: Create `code/extraction/test_ir_extractor.py`
- [x] Step 3.2: Verify extraction is deterministic
- [x] Step 3.3: Verify extracted IR is valid
- **Done when**: Test suite passes twice ✓

## Task 4: Document IR format
- [x] Step 4.1: Create `notes/ir_format.md`
- [x] Step 4.2: Document HLO structure and key ops
- **Done when**: Notes file exists and is under 30 lines ✓
