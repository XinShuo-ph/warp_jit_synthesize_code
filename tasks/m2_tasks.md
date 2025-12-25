# Milestone 2 Tasks

## Task 1: IR Extractor Design
- [x] Step 1.1: Analyze requirements for robust IR extraction
- [x] Step 1.2: Design API for ir_extractor.py
- [x] Step 1.3: Handle edge cases (generic kernels, device variations)
- **Done when**: Clear design documented in code comments ✓

## Task 2: Implement IR Extractor
- [x] Step 2.1: Implement `extract_ir(kernel, device)` function
- [x] Step 2.2: Add module compilation and cache lookup
- [x] Step 2.3: Add error handling and validation
- **Done when**: Function successfully extracts IR from any kernel ✓

## Task 3: Create Test Cases
- [x] Step 3.1: Test case 1 - Simple arithmetic kernel
- [x] Step 3.2: Test case 2 - Vector operations
- [x] Step 3.3: Test case 3 - Control flow (if/else)
- [x] Step 3.4: Test case 4 - Loops (for/while)
- [x] Step 3.5: Test case 5 - Functions and nested calls
- **Done when**: 5+ kernels with extracted IR saved to files ✓

## Task 4: Analyze IR Structure
- [x] Step 4.1: Identify common patterns in generated C++
- [x] Step 4.2: Document SSA variable naming scheme
- [x] Step 4.3: Document function signatures and conventions
- [x] Step 4.4: Document adjoint (backward) pass structure
- **Done when**: notes/ir_format.md documents IR structure ✓

## Task 5: Create Paired Dataset Samples
- [x] Step 5.1: Save Python source + IR pairs as structured data
- [x] Step 5.2: Include metadata (kernel name, device, hash)
- [x] Step 5.3: Validate data quality
- **Done when**: 5+ samples in data/ directory with proper format ✓

## Validation
- [x] IR extractor works on all test kernels
- [x] Extraction runs twice with identical results
- [x] Documentation is complete and accurate
- [x] All code is clean (no debug prints)

## Milestone 2 Complete ✓
All deliverables achieved:
- `code/extraction/ir_extractor.py`: Robust extraction function
- 5 test cases with Python→IR pairs in `data/test_cases/`
- `notes/ir_format.md`: IR structure documentation
