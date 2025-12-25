# Milestone 3 Tasks

## Task 1: Explore Warp FEM
- [ ] Step 1.1: Read `warp.fem` documentation (if available) or source code structure.
- [ ] Step 1.2: Run a basic existing FEM example if one exists in the repo/install.
- **Done when**: I have a basic understanding of `wp.fem` classes (`Grid`, `Field`, `Operator`).

## Task 2: Implement Poisson Solver
- [ ] Step 2.1: Create `jit/code/examples/poisson_solver.py`.
- [ ] Step 2.2: Set up a 2D grid.
- [ ] Step 2.3: Define trial/test functions.
- [ ] Step 2.4: Assemble stiffness matrix and RHS for $\Delta u = f$.
- [ ] Step 2.5: Solve the linear system.
- **Done when**: The script runs and solves the system.

## Task 3: Validate Solver
- [ ] Step 3.1: Create `jit/code/examples/test_poisson.py`.
- [ ] Step 3.2: Compare numerical result against analytical solution for a simple case (e.g., $f = 2\pi^2 \sin(\pi x)\sin(\pi y)$).
- **Done when**: Test passes with reasonable error tolerance.
