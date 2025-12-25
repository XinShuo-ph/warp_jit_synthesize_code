# Milestone 3 Tasks

## Task 1: Study warp.fem API
- [x] Step 1.1: Review example_diffusion.py
- [x] Step 1.2: Review fem/utils.py for mesh generation and solvers
- **Done when**: Understand @fem.integrand, fem.integrate, fem.make_test/trial ✓

## Task 2: Implement Poisson Solver
- [x] Step 2.1: Create poisson_solver.py with basic structure
- [x] Step 2.2: Define weak form integrands (Laplacian, source)
- [x] Step 2.3: Set up mesh, function spaces, boundary conditions
- [x] Step 2.4: Solve linear system with CG solver
- **Done when**: Solver runs without errors ✓

## Task 3: Validation Tests
- [x] Step 3.1: Create test_poisson.py with 4 tests
- [x] Step 3.2: Compare to analytical solution (u = sin(πx)sin(πy))
- [x] Step 3.3: Verify tests pass twice (4/4 both runs)
- **Done when**: Tests pass for 2 consecutive runs ✓
