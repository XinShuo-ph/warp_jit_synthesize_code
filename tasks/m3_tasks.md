# Milestone 3 Tasks

## Task 1: Study warp.fem Module
- [x] Step 1.1: Review warp.fem API documentation
- [x] Step 1.2: Study example_diffusion.py structure
- [x] Step 1.3: Understand integrand, Field, Sample concepts
- [x] Step 1.4: Understand geometry and space creation
- **Done when**: Can explain FEM workflow in warp ✓

## Task 2: Implement Poisson Solver
- [x] Step 2.1: Create basic 2D Poisson solver structure
- [x] Step 2.2: Implement weak form integrands
- [x] Step 2.3: Set up boundary conditions (Dirichlet)
- [x] Step 2.4: Assemble and solve linear system
- **Done when**: Solver runs and produces output ✓

## Task 3: Add Analytical Solutions
- [x] Step 3.1: Implement analytical solution for known case
- [x] Step 3.2: Add comparison/error computation
- [x] Step 3.3: Test multiple domain sizes
- **Done when**: Can compute L2 error vs analytical solution ✓

## Task 4: Create Validation Tests
- [x] Step 4.1: Test 1 - Constant forcing, zero BC
- [x] Step 4.2: Test 2 - Manufactured solution
- [x] Step 4.3: Verify convergence with refinement
- [x] Step 4.4: Run tests twice, verify consistency
- **Done when**: test_poisson.py passes all tests twice ✓

## Task 5: Document and Clean
- [x] Step 5.1: Add docstrings to all functions
- [x] Step 5.2: Remove debug code
- [x] Step 5.3: Add usage example
- **Done when**: Code is clean and documented ✓

## Validation Criteria
- [x] Poisson solver completes successfully
- [x] Tests compare to analytical solutions
- [x] L2 error is reasonable (< 1e-3 for moderate resolution)
- [x] Tests pass 2+ consecutive times with identical results
- [x] Code is clean and documented

## Milestone 3 Complete ✓
All deliverables achieved:
- `code/examples/poisson_solver.py`: Working 2D Poisson solver with manufactured solution
- `code/examples/test_poisson.py`: Comprehensive test suite with 5 tests
- All tests pass consistently across multiple runs
- Solution uses warp.fem with proper boundary conditions
