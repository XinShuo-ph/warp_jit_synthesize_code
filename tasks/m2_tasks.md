# Milestone 2 Tasks

## Task 1: Create IR Extractor Utility
- [x] Step 1.1: Design API for ir_extractor.py
- [x] Step 1.2: Implement extract_ir() function
- [x] Step 1.3: Add helper functions for parsing C++ code
- [x] Step 1.4: Add support for extracting forward/backward functions separately
- **Done when**: `code/extraction/ir_extractor.py` can extract IR from any kernel

## Task 2: Create Test Cases
- [x] Step 2.1: Create 5 diverse test kernels (arithmetic, vectors, control flow, loops, atomics)
- [x] Step 2.2: Extract IR for each kernel
- [x] Step 2.3: Save Python→IR pairs in structured format
- [x] Step 2.4: Verify extraction works consistently
- **Done when**: 5+ test cases with validated Python→IR pairs

## Task 3: Document IR Structure
- [x] Step 3.1: Analyze IR patterns across test cases
- [x] Step 3.2: Document variable naming conventions
- [x] Step 3.3: Document function structure
- [x] Step 3.4: Create ir_format.md (max 30 lines)
- **Done when**: `notes/ir_format.md` describes IR structure comprehensively

## Task 4: Verification
- [x] Step 4.1: Run extractor on all test cases twice
- [x] Step 4.2: Verify consistent extraction
- **Done when**: All tests pass twice with identical results
