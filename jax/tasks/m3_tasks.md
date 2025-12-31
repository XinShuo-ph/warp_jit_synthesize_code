# Milestone 3 Tasks: Numerical Computing - Poisson Solver

## Task 1: Implement Poisson solver
- [x] Step 1.1: Implement 2D Laplacian operator using finite differences
- [x] Step 1.2: Implement Jacobi iterative solver
- [x] Step 1.3: JIT compile the solver for performance
- **Done when**: Solver converges for test problem ✓

## Task 2: Validate against analytical solutions
- [x] Step 2.1: Test case 1: u = sin(πx)sin(πy) → f = -2π²sin(πx)sin(πy)
- [x] Step 2.2: Test case 2: u = x² + y² → f = 4
- [x] Step 2.3: Compare numerical vs analytical, error < 1e-3
- **Done when**: Tests pass for both cases ✓

## Task 3: Create test file
- [x] Step 3.1: Create `code/examples/test_poisson.py`
- [x] Step 3.2: Run tests twice
- **Done when**: Tests pass twice consecutively ✓
