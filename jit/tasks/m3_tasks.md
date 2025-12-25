# Milestone 3 Tasks: FEM Deep Dive

## Task 1: Explore FEM Module
- [x] Step 1.1: Run `warp/examples/fem/example_diffusion.py` (or similar) to ensure FEM tools work.
- [x] Step 1.2: Analyze the structure of a Warp FEM application (Geometry, Space, Field, Integrators).
- **Done when**: A standard FEM example runs successfully and its components are understood.

## Task 2: Implement Poisson Solver
- [x] Step 2.1: Create `jit/code/examples/poisson_solver.py`.
- [x] Step 2.2: Define a 2D Grid domain.
- [x] Step 2.3: Implement the weak form of the Poisson equation ($\nabla^2 u = f$).
- [x] Step 2.4: Solve the system using `warp.fem` utilities (likely CG or direct solver if available).
- **Done when**: `poisson_solver.py` runs and produces a solution field.

## Task 3: Validation
- [x] Step 3.1: Create `jit/code/examples/test_poisson.py`.
- [x] Step 3.2: Implement an analytical solution comparison (Method of Manufactured Solutions).
    - e.g., Set $u_{exact} = \sin(\pi x) \sin(\pi y)$, derive $f$, solve, compare error.
- [x] Step 3.3: Ensure L2 error decreases with grid resolution (convergence test).
- **Done when**: The solver is mathematically verified with tests passing.
