# Milestone 2 Tasks: CUDA IR Extraction

## Task 2.1: Create CUDA IR Extractor
- [ ] Copy ir_extractor.py to cuda_ir_extractor.py
- [ ] Verify device parameter works with "cuda"
- [ ] Test extraction with simple_add kernel
- [ ] Verify CUDA-specific code patterns are captured
- **Done when**: cuda_ir_extractor.py extracts CUDA IR successfully

## Task 2.2: Test Each Kernel Type (Arithmetic)
- [ ] Generate arithmetic kernel
- [ ] Extract CUDA IR
- [ ] Verify function signature has __global__
- [ ] Verify grid-stride loop present
- [ ] Save sample pair
- **Done when**: Arithmetic kernel generates valid CUDA IR

## Task 2.3: Test Vector Operations
- [ ] Generate vector kernel (vec3 dot product)
- [ ] Extract CUDA IR
- [ ] Verify vector operations in CUDA code
- [ ] Save sample pair
- **Done when**: Vector kernel generates valid CUDA IR

## Task 2.4: Test Matrix Operations
- [ ] Generate matrix kernel (mat-vec multiply)
- [ ] Extract CUDA IR
- [ ] Verify matrix operations in CUDA code
- [ ] Save sample pair
- **Done when**: Matrix kernel generates valid CUDA IR

## Task 2.5: Test Math Functions
- [ ] Generate math kernel (sin, cos, exp)
- [ ] Extract CUDA IR
- [ ] Verify CUDA math functions used
- [ ] Save sample pair
- **Done when**: Math kernel generates valid CUDA IR

## Task 2.6: Test Control Flow
- [ ] Generate control flow kernel (if/else and loops)
- [ ] Extract CUDA IR
- [ ] Verify branches in CUDA code
- [ ] Save sample pair
- **Done when**: Control flow kernel generates valid CUDA IR

## Task 2.7: Test Atomic Operations
- [ ] Generate atomic kernel
- [ ] Extract CUDA IR
- [ ] Verify atomic operations in CUDA code
- [ ] Save sample pair
- **Done when**: Atomic kernel generates valid CUDA IR

## Task 2.8: Create Automated Test Suite
- [ ] Create test_cuda_extraction.py
- [ ] Test all 6 kernel types
- [ ] Verify CUDA-specific patterns in each
- [ ] Generate 10+ test pairs
- **Done when**: Test suite passes for all kernel types

## Task 2.9: Document CUDA IR Format
- [ ] Document CUDA function signature format
- [ ] Document grid-stride loop pattern
- [ ] Document differences from CPU IR
- [ ] Create notes/cuda_ir_format.md
- **Done when**: Documentation complete with examples
