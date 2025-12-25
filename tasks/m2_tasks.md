# Milestone 2 Tasks

## Task 1: Design IR Extraction API
- [x] Step 1.1: Define function signature for `extract_ir(kernel) -> dict`
- [x] Step 1.2: Determine what metadata to extract (kernel name, args, source, etc.)
- [x] Step 1.3: Design output format for Python→IR pairs
- **Done when**: Clear API design documented in ir_extractor.py

## Task 2: Implement Basic IR Extractor
- [x] Step 2.1: Create `code/extraction/ir_extractor.py` 
- [x] Step 2.2: Implement function to trigger kernel compilation
- [x] Step 2.3: Implement function to locate cache file for compiled kernel
- [x] Step 2.4: Implement function to read and parse C++ IR from cache
- **Done when**: Can extract IR from a simple kernel

## Task 3: Extract Python Source Code
- [x] Step 3.1: Use inspect module to get Python source
- [x] Step 3.2: Extract function signature and body
- [x] Step 3.3: Format Python code for pairing with IR
- **Done when**: Can extract both Python and IR from same kernel

## Task 4: Create Test Cases
- [x] Step 4.1: Create 6 diverse test kernels (arithmetic, indexing, conditional, loop, vector, complex)
- [x] Step 4.2: Extract Python→IR pairs for each test
- [x] Step 4.3: Save pairs in structured format (JSON)
- [x] Step 4.4: Verify pairs are correct and complete
- **Done when**: 5+ validated Python→IR pairs exist

## Task 5: Document IR Format
- [x] Step 5.1: Analyze structure of generated IR
- [x] Step 5.2: Document variable naming conventions
- [x] Step 5.3: Document operation mapping (Python ops → C++ calls)
- [x] Step 5.4: Create notes/ir_format.md (<30 lines)
- **Done when**: IR format is documented and understandable

## Status: COMPLETED
All tasks completed. Created ir_extractor.py with extract_ir() and extract_ir_pair() functions.
Generated 6 test cases saved to data/test_cases.json. Ready for M3.
