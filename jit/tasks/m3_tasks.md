# Milestone 3 Tasks

## Task 1: FEM Poisson Solver Implementation
- [x] Step 1.1: Study `warp/examples/fem/example_diffusion.py` (which is similar to Poisson).
- [x] Step 1.2: Create `code/examples/poisson_solver.py` implementing Poisson equation $-\Delta u = f$.
- [x] Step 1.3: Define a known analytical solution for validation (e.g., $u = \sin(\pi x) \sin(\pi y)$).
- **Done when**: `poisson_solver.py` runs and solves the system.

## Task 2: Validation Tests
- [x] Step 2.1: Create `code/examples/test_poisson.py`.
- [x] Step 2.2: Compute L2 error against analytical solution.
- [x] Step 2.3: Verify error convergence or small magnitude.
- **Done when**: Tests pass with acceptable error margin.
