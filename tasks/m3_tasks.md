# Milestone 3 Tasks

## Task 1: Study warp.fem Module
- [x] Step 1.1: Locate and read warp.fem documentation
- [x] Step 1.2: Study FEM examples in warp repo
- [x] Step 1.3: Understand FEM primitives (elements, spaces, integrals)
- **Done when**: Can explain key warp.fem concepts

## Task 2: Implement Poisson Solver
- [x] Step 2.1: Set up domain and boundary conditions
- [x] Step 2.2: Define weak formulation with warp.fem
- [x] Step 2.3: Assemble system and solve
- [x] Step 2.4: Visualize/save results
- **Done when**: `code/examples/poisson_solver.py` runs successfully

## Task 3: Create Validation Tests
- [x] Step 3.1: Define analytical test cases (e.g., u = sin(pi*x)*sin(pi*y))
- [x] Step 3.2: Implement test_poisson.py with convergence checks
- [x] Step 3.3: Verify numerical solution matches analytical
- **Done when**: `code/examples/test_poisson.py` passes with L2 error < 1e-3

## Task 4: Verification
- [x] Step 4.1: Run tests twice
- [x] Step 4.2: Verify consistent results
- **Done when**: Tests pass 2+ times with same results
