# Milestone 3 Tasks: Scientific Computing Deep Dive

## Task 1: Design Poisson Solver Architecture
- [ ] Step 1.1: Choose numerical method (Jacobi iteration or direct solve)
- [ ] Step 1.2: Design JAX-compatible solver with JIT compilation
- [ ] Step 1.3: Plan boundary conditions handling
- **Done when**: Clear design for solver implementation

## Task 2: Implement Poisson Solver
- [ ] Step 2.1: Implement 1D Poisson equation solver
- [ ] Step 2.2: Implement 2D Poisson equation solver
- [ ] Step 2.3: Add boundary condition options (Dirichlet, Neumann)
- [ ] Step 2.4: Ensure all functions are JIT-compatible
- **Done when**: code/examples/poisson_solver.py runs successfully

## Task 3: Create Analytical Test Cases
- [ ] Step 3.1: Define analytical test case 1: sin(x) forcing
- [ ] Step 3.2: Define analytical test case 2: polynomial forcing
- [ ] Step 3.3: Implement analytical solution calculator
- **Done when**: Have 2+ test cases with known solutions

## Task 4: Implement Validation Tests
- [ ] Step 4.1: Create test_poisson.py with pytest structure
- [ ] Step 4.2: Compare numerical solution to analytical solution
- [ ] Step 4.3: Assert L2 error < tolerance (e.g., 1e-3)
- [ ] Step 4.4: Test on different grid sizes
- **Done when**: All tests pass with good accuracy

## Task 5: Run Validation Suite
- [ ] Step 5.1: Run tests twice to ensure consistency
- [ ] Step 5.2: Verify results match both times
- [ ] Step 5.3: Document any numerical precision issues
- **Done when**: Tests pass 2 consecutive times with same results

## Success Criteria for M3
- [ ] code/examples/poisson_solver.py exists and works
- [ ] code/examples/test_poisson.py exists with 2+ test cases
- [ ] All tests pass with L2 error < 1e-3
- [ ] Tests run twice with consistent results
- [ ] Solver uses JAX JIT compilation
