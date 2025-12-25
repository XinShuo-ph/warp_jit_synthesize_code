# Milestone 2 Tasks

## Task 1: Enhance IR extractor with robust features
- [x] Step 1.1: Add error handling for missing/corrupted cache files
- [x] Step 1.2: Add validation that Python and C++ correspond
- [x] Step 1.3: Add option to clean/rebuild cache
- [x] Step 1.4: Test with edge cases (empty kernel, complex types)
- **Done when**: Extractor handles errors gracefully and validates output

## Task 2: Create batch extraction utility
- [x] Step 2.1: Function to extract IR from multiple kernels at once
- [x] Step 2.2: Progress reporting for batch operations
- [x] Step 2.3: Parallel extraction support (if beneficial)
- **Done when**: Can extract 10+ kernels efficiently

## Task 3: Generate diverse test cases
- [x] Step 3.1: Create 5+ additional test kernels (total 10+)
- [x] Step 3.2: Include: structs, while loops, nested conditionals, math functions
- [x] Step 3.3: Save all Python→IR pairs to data/samples/
- **Done when**: 10+ diverse test cases successfully extracted

## Task 4: Document IR format
- [x] Step 4.1: Analyze C++ IR structure in detail
- [x] Step 4.2: Document variable naming conventions
- [x] Step 4.3: Document how Python constructs map to C++
- [x] Step 4.4: Create notes/ir_format.md (max 30 lines)
- **Done when**: Documentation clearly explains IR structure

## Task 5: Validation
- [x] Step 5.1: Run all extraction tests twice
- [x] Step 5.2: Verify IR is identical across runs
- [x] Step 5.3: Verify no data corruption or missing fields
- **Done when**: All tests pass validation protocol
