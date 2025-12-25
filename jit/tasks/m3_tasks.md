# Milestone 3 Tasks (FEM Deep Dive)

## Task 1: Study Warp FEM patterns
- [x] Step 1.1: Locate Warp FEM examples in the installed package
- [x] Step 1.2: Identify the minimal API set for: mesh, function space, forms, boundary conditions, linear solve
- **Done when**: We can point to a working example pattern that assembles and solves a Poisson-like problem.

## Task 2: Implement Poisson solver example
- [x] Step 2.1: Add `jit/code/examples/poisson_solver.py` with a function `solve_poisson_unit_square(...)`
- [x] Step 2.2: Use an analytic solution on [0,1]^2 with Dirichlet BC (e.g., sin(pi x) sin(pi y))
- [x] Step 2.3: Run on CPU and return solution field and basic error norms
- **Done when**: Script runs end-to-end and reports bounded error for a reasonable resolution.

## Task 3: Add validation tests
- [x] Step 3.1: Add `jit/code/examples/test_poisson.py` running 2+ resolutions and checking error decreases
- [x] Step 3.2: Ensure tests run twice with identical results (no randomness)
- **Done when**: `python3 -m jit.code.examples.test_poisson` passes twice consecutively.

