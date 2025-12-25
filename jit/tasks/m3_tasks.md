# Milestone 3 Tasks: FEM Deep Dive - Poisson Solver

## Task 1: Study warp.fem Framework
- [x] Step 1.1: Understand @fem.integrand decorator
- [x] Step 1.2: Understand fem.make_test/trial, fem.integrate
- [x] Step 1.3: Understand boundary condition handling
- **Done when**: Can explain FEM workflow in warp ✓

## Task 2: Implement Poisson Solver
- [x] Step 2.1: Create poisson_solver.py with 2D Poisson equation
- [x] Step 2.2: Implement -∇²u = f with Dirichlet BCs
- [x] Step 2.3: Use Grid2D mesh with configurable resolution
- **Done when**: Solver runs without errors ✓

## Task 3: Create Validation Tests
- [x] Step 3.1: Test with known analytical solution (u=sin(πx)sin(πy))
- [x] Step 3.2: Compute L2 error norm
- [x] Step 3.3: Verify convergence with mesh refinement
- **Done when**: Tests pass for 2+ consecutive runs ✓ (4/4 tests, 2 runs)

## Analytical Test Case
For -∇²u = f on [0,1]×[0,1] with u=0 on boundary:
- Choose u(x,y) = sin(πx)sin(πy)
- Then f(x,y) = 2π²sin(πx)sin(πy)
- This provides exact solution for validation
