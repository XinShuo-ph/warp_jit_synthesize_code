# Milestone 4 Tasks: Synthesis Pipeline

## Task 1: Design Kernel Generator Architecture
- [ ] Step 1.1: Identify kernel types to generate (arithmetic, array, math, linalg, control flow)
- [ ] Step 1.2: Design template-based generation system
- [ ] Step 1.3: Plan parameter variation strategy (shapes, dtypes, operations)
- **Done when**: Clear architecture for generator.py

## Task 2: Implement Kernel Generator
- [ ] Step 2.1: Create KernelGenerator class with generate_* methods
- [ ] Step 2.2: Implement arithmetic kernel generation (add, sub, mul, div, power)
- [ ] Step 2.3: Implement array operation generation (reshape, transpose, slice, concat)
- [ ] Step 2.4: Implement math function generation (sin, cos, exp, log, tanh, etc.)
- [ ] Step 2.5: Implement reduction generation (sum, mean, max, min with axis options)
- [ ] Step 2.6: Implement linear algebra generation (dot, matmul, outer)
- [ ] Step 2.7: Add control flow generation (cond, scan, while_loop)
- **Done when**: generator.py creates diverse valid JAX functions

## Task 3: Implement Synthesis Pipeline
- [ ] Step 3.1: Create SynthesisPipeline class
- [ ] Step 3.2: Implement generate_kernel → extract_ir → create_pair workflow
- [ ] Step 3.3: Add error handling for invalid kernels
- [ ] Step 3.4: Implement batch generation method
- [ ] Step 3.5: Add progress tracking and logging
- **Done when**: pipeline.py can generate pairs end-to-end

## Task 4: Test Pipeline Components
- [ ] Step 4.1: Test generator creates valid Python code
- [ ] Step 4.2: Test generated functions execute without errors
- [ ] Step 4.3: Test IR extraction succeeds for generated kernels
- [ ] Step 4.4: Verify training pairs have all required fields
- **Done when**: Pipeline runs without errors on test cases

## Task 5: Generate Sample Dataset
- [ ] Step 5.1: Run pipeline to generate 100+ pairs
- [ ] Step 5.2: Save to data/samples/ directory
- [ ] Step 5.3: Verify all pairs are valid and loadable
- [ ] Step 5.4: Check diversity (multiple categories represented)
- **Done when**: data/samples/ contains 100+ valid diverse pairs

## Success Criteria for M4
- [ ] code/synthesis/generator.py exists and works
- [ ] code/synthesis/pipeline.py exists and works
- [ ] Can generate 7+ types of kernels
- [ ] data/samples/ contains 100+ valid training pairs
- [ ] Pairs cover multiple categories (arithmetic, array, math, linalg, etc.)
- [ ] All generated code is valid and executable
