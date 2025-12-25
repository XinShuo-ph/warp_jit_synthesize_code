# Milestone 4 Tasks

## Task 1: Kernel Generator
- [x] Step 1.1: Create generator.py with kernel template system
- [x] Step 1.2: Implement variations: ops, types, loop counts, branches
- [x] Step 1.3: Ensure generated kernels are valid (compile test)
- **Done when**: Generator produces 10+ unique valid kernels ✓

## Task 2: Synthesis Pipeline
- [x] Step 2.1: Create pipeline.py connecting generator → extractor
- [x] Step 2.2: Add output formatting (JSON with python/cpp pairs)
- [x] Step 2.3: Add batch generation support
- **Done when**: Pipeline generates pairs end-to-end ✓

## Task 3: Generate Sample Data
- [x] Step 3.1: Generate 110 varied kernel pairs (0 failures)
- [x] Step 3.2: Save to data/samples/training_pairs.json (620KB)
- [x] Step 3.3: Validate: 10 generator types, avg 186 chars py, 5134 chars cpp
- **Done when**: 100+ pairs in data/samples/ ✓
