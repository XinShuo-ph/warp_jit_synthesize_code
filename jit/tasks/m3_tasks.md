# Milestone 3: Scientific Computing Example (Poisson Solver)

## Task 1: Implement 2D Poisson Solver in JAX
- [x] Create `jit/code/examples/poisson_solver.py`.
- [x] Implement Finite Difference Method (FDM) using JAX array operations (convolutions or shifts).
- [x] Use `jax.jit` for performance.
- **Done when**: Solver runs and produces output.

## Task 2: Validation
- [x] Create `jit/code/examples/test_poisson.py`.
- [x] Compare numerical result against analytical solution for a simple case (e.g., sin(x)sin(y)).
- **Done when**: Error is within acceptable bounds (e.g., < 1e-3).

## Task 3: IR Extraction for Solver
- [x] Extract IR for the core solver step.
- [x] Save to `jit/data/poisson_step.hlo`.
- **Done when**: IR file exists.
