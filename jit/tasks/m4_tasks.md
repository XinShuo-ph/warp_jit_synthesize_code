# Milestone 4 Tasks

## Task 1: Create Kernel Generator
- [ ] Step 1.1: Generate arithmetic kernels (add, mul, sub, div)
- [ ] Step 1.2: Generate math function kernels (sin, cos, exp, tanh)
- [ ] Step 1.3: Generate array operation kernels (dot, matmul, reduce)
- [ ] Step 1.4: Generate control flow kernels (where, cond)
- [ ] Step 1.5: Generate combined/complex kernels
- **Done when**: Generator produces varied JAX functions programmatically

## Task 2: Create Synthesis Pipeline
- [ ] Step 2.1: Generate function → extract IR → save pair
- [ ] Step 2.2: Include metadata (function type, complexity)
- [ ] Step 2.3: Handle errors gracefully
- [ ] Step 2.4: Support batch generation
- **Done when**: Pipeline generates valid Python→IR pairs end-to-end

## Task 3: Validation
- [ ] Step 3.1: Validate generated functions execute correctly
- [ ] Step 3.2: Validate IR extraction succeeds
- [ ] Step 3.3: Validate JSON format
- **Done when**: 100+ sample pairs generated and validated

## Task 4: Testing
- [ ] Step 4.1: Test each kernel type generator
- [ ] Step 4.2: Test pipeline with various parameters
- [ ] Step 4.3: Verify output quality
- **Done when**: Tests pass consistently
