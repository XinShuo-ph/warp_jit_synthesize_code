# Milestone 3 Tasks

## Task 1: Choose Poisson problem + analytic solution
- [x] Step 1.1: Pick domain + BCs (unit square, full Dirichlet)
- [x] Step 1.2: Pick analytic solution `u(x,y)` and derive forcing `f(x,y)` for `-Δu = f`
- **Done when**: `u` and `f` are fixed and implemented as FEM integrands.

## Task 2: Implement solver
- [x] Step 2.1: Create `jit/code/examples/poisson_solver.py`
- [x] Step 2.2: Assemble stiffness matrix `∫ grad(u)·grad(v)` and RHS `∫ f v`
- [x] Step 2.3: Apply Dirichlet BCs by nodal projection on boundary nodes
- [x] Step 2.4: Solve with CG (via existing FEM example utilities)
- **Done when**: Solver runs on CPU and returns a discrete field solution.

## Task 3: Validation tests
- [x] Step 3.1: Create `jit/code/examples/test_poisson.py` (pytest)
- [x] Step 3.2: Compare numeric solution to analytic (L2 or max-norm at nodes) for 2+ resolutions
- [x] Step 3.3: Run tests twice consecutively
- **Done when**: Tests pass twice and error decreases with increased resolution.

