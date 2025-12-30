# Milestone 4 Tasks: Synthesis Pipeline

## Task 1: Build function generator
- [x] Step 1.1: Create `generator.py` with template-based function generation
- [x] Step 1.2: Support multiple operation types (arithmetic, trig, reduction, etc.)
- [x] Step 1.3: Support varying input shapes and types
- **Done when**: Can generate 100+ unique valid functions

## Task 2: Build pipeline
- [x] Step 2.1: Create `pipeline.py` that combines generator + extractor
- [x] Step 2.2: Generate functions, extract IR, save pairs
- [x] Step 2.3: Handle errors gracefully
- **Done when**: Pipeline produces 100+ sample pairs

## Task 3: Validate output
- [x] Step 3.1: Verify all generated pairs are valid
- [x] Step 3.2: Check for diversity in operations
- **Done when**: 100+ valid pairs in data/samples/
