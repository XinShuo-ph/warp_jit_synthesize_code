# Milestone 2 Tasks: IR Extraction Mechanism

## Task 1: Create IR Extractor Module
- [x] Step 1.1: Design IRExtractor class with extract_jaxpr and extract_stablehlo methods
- [x] Step 1.2: Implement create_training_pair method
- [x] Step 1.3: Add save_pair and save_pairs methods for JSON export
- **Done when**: ir_extractor.py runs and creates valid training pairs

## Task 2: Test IR Extraction on Multiple Function Types
- [x] Step 2.1: Test arithmetic operations (add, sub, mul, div)
- [x] Step 2.2: Test array operations (reshape, transpose, slice, concat)
- [x] Step 2.3: Test math functions (sin, cos, exp, log, sqrt)
- [x] Step 2.4: Test reduction operations (sum, mean, max, min)
- [x] Step 2.5: Test linear algebra (dot, matmul, outer)
- **Done when**: test_ir_extractor.py passes with 20+ test cases

## Task 3: Validate IR Extraction Quality
- [x] Step 3.1: Verify extracted Jaxpr is parseable and complete
- [x] Step 3.2: Verify extracted StableHLO is valid MLIR
- [x] Step 3.3: Ensure same function produces same IR (deterministic)
- **Done when**: Can reload saved pairs and all contain valid IR

## Task 4: Document IR Formats
- [x] Step 4.1: Document Jaxpr structure and type notation
- [x] Step 4.2: Document StableHLO structure and common operations
- [x] Step 4.3: Create notes/ir_format.md (max 30 lines)
- **Done when**: notes/ir_format.md exists and covers both IR types

## Task 5: Create Sample Dataset
- [x] Step 5.1: Generate 20+ diverse Python→IR pairs
- [x] Step 5.2: Save to data/samples/test_cases.json
- [x] Step 5.3: Verify all pairs are loadable
- **Done when**: data/samples/ contains valid JSON with 20+ pairs

## Success Criteria for M2
- [x] ir_extractor.py implemented and working
- [x] Can extract both Jaxpr and StableHLO
- [x] 23 test cases covering 5 categories
- [x] notes/ir_format.md exists and is accurate
- [x] data/samples/test_cases.json contains 23 valid pairs
- [x] All tests run twice with identical results
