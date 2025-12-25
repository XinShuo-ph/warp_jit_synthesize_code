# Milestone 4 Tasks

## Task 1: Create Kernel Generator
- [x] Step 1.1: Design generator API and kernel templates
- [x] Step 1.2: Implement basic arithmetic kernel generator
- [x] Step 1.3: Add vector and matrix operation generators
- [x] Step 1.4: Add control flow generators (if/else, loops)
- **Done when**: `code/synthesis/generator.py` can generate diverse kernels

## Task 2: Build End-to-End Pipeline
- [x] Step 2.1: Create pipeline.py skeleton
- [x] Step 2.2: Integrate generator → compilation → extraction
- [x] Step 2.3: Add error handling and validation
- [x] Step 2.4: Save pairs in structured format
- **Done when**: `code/synthesis/pipeline.py` produces Python→IR pairs

## Task 3: Generate Sample Dataset
- [x] Step 3.1: Run pipeline to generate 100+ samples
- [x] Step 3.2: Verify all samples are valid
- [x] Step 3.3: Check diversity of generated kernels
- **Done when**: `data/samples/` contains 100+ valid pairs

## Task 4: Verification
- [x] Step 4.1: Sample and inspect generated pairs
- [x] Step 4.2: Verify IR extraction quality
- [x] Step 4.3: Test pipeline twice for consistency
- **Done when**: Pipeline runs reliably with consistent output
