# Milestone 4 Tasks

## Task 1: Design Kernel Generator
- [x] Step 1.1: Define kernel generation patterns (arithmetic, array ops, reductions, etc.)
- [x] Step 1.2: Create parameter spaces for variation (operators, constants, array sizes)
- [x] Step 1.3: Design template system for kernel generation
- [x] Step 1.4: Plan diversity metrics (ensure variety in generated kernels)
- **Done when**: Clear design for generating diverse kernels

## Task 2: Implement Kernel Generator
- [x] Step 2.1: Create `code/synthesis/generator.py`
- [x] Step 2.2: Implement arithmetic kernel generator
- [x] Step 2.3: Implement array operation generator
- [x] Step 2.4: Implement control flow generator (if/for)
- [x] Step 2.5: Implement vector operation generator
- [x] Step 2.6: Add randomization and variation
- **Done when**: Can generate 10+ diverse kernel patterns

## Task 3: Implement Synthesis Pipeline
- [x] Step 3.1: Create `code/synthesis/pipeline.py`
- [x] Step 3.2: Integrate kernel generator with IR extractor
- [x] Step 3.3: Add compilation and extraction logic
- [x] Step 3.4: Implement data saving (JSON format)
- [x] Step 3.5: Add error handling and logging
- **Done when**: End-to-end pipeline works for single kernel

## Task 4: Generate Sample Dataset
- [x] Step 4.1: Run pipeline to generate 120 samples
- [x] Step 4.2: Verify all samples are valid
- [x] Step 4.3: Check for diversity in samples
- [x] Step 4.4: Save to `data/samples/`
- **Done when**: 100+ valid Python→IR pairs in data/samples/

## Task 5: Validate and Document
- [x] Step 5.1: Spot-check sample quality
- [x] Step 5.2: Verify IR correctness
- [x] Step 5.3: Add usage documentation to pipeline.py
- [x] Step 5.4: Run pipeline twice to verify reproducibility
- **Done when**: Dataset is validated and ready for M5

## Status: COMPLETED
Generated 120 Python→IR pairs (239KB JSON file).
Avg Python length: 260 chars, Avg C++ IR: 1528 chars.
7 kernel types: arithmetic, array indexing, conditional, loop, vector, math, multi-op.
