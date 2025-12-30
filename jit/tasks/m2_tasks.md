# Milestone 2 Tasks

## Task 1: Create IR Extractor Utility
- [ ] Step 1.1: Create ir_extractor.py with extract_ir() function
- [ ] Step 1.2: Support both HLO and StableHLO formats
- [ ] Step 1.3: Include metadata (function name, input shapes, dtypes)
- **Done when**: Can extract IR from any JAX function

## Task 2: Create Test Cases
- [ ] Step 2.1: Test arithmetic operations (add, mul, sub, div)
- [ ] Step 2.2: Test math functions (sin, cos, exp, log, tanh)
- [ ] Step 2.3: Test array operations (dot, matmul, sum, transpose)
- [ ] Step 2.4: Test control flow (where, cond, scan)
- [ ] Step 2.5: Test vectorization (vmap)
- **Done when**: 5+ test cases with Python→IR pairs

## Task 3: Document IR Format
- [ ] Step 3.1: Analyze HLO vs StableHLO structure
- [ ] Step 3.2: Document common operations
- [ ] Step 3.3: Note type representations
- **Done when**: notes/ir_format.md complete (max 30 lines)

## Task 4: Save Sample Pairs
- [ ] Step 4.1: Create save_sample_pairs.py utility
- [ ] Step 4.2: Save as JSON with python_source and ir_code
- [ ] Step 4.3: Generate 5+ samples in data/
- **Done when**: Sample JSON files exist and are valid
