# Milestone 3 Tasks: Numerical Solver Deep Dive

## Task 1: Poisson Solver Implementation
- [x] Step 1.1: Create `code/examples/poisson_solver.py`.
- [x] Step 1.2: Implement `solve_poisson(f, n_iter)` using Jacobi iteration.
- [x] Step 1.3: Use `jax.lax.scan` or `jax.jit` loop for iterations to ensure it's compile-able.
- **Done when**: The solver runs and produces a solution grid.

## Task 2: Validation
- [x] Step 2.1: Create `code/examples/test_poisson.py`.
- [x] Step 2.2: Test with a known analytical solution (e.g., $u = \sin(x)\sin(y)$).
- [x] Step 2.3: Verify error decreases with iterations.
- **Done when**: Tests pass showing convergence to known solution.

## Task 3: IR Analysis
- [x] Step 3.1: Extract IR for the `step` function and the full `solve` loop.
- [x] Step 3.2: Verify the IR size is reasonable (not unrolling the loop into thousands of ops).
- **Done when**: We have confirmed `jax.lax.scan` or `while_loop` preserves loop structure in IR.
