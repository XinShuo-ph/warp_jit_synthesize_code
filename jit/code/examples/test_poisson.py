"""Validation tests for Poisson solver."""
import sys
import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})
wp.init()

from poisson_solver import PoissonSolver, compute_analytical_solution, compute_l2_error


def test_convergence_with_resolution():
    """Test that error decreases with mesh refinement."""
    print("\n" + "=" * 60)
    print("Test 1: Convergence with mesh refinement")
    print("=" * 60)
    
    resolutions = [10, 20, 40]
    errors = []
    
    for res in resolutions:
        solver = PoissonSolver(resolution=res, degree=1)
        field = solver.solve(source_type="sin", bc_value=0.0)
        
        positions = solver.get_dof_positions()
        u_num = field.dof_values.numpy()
        u_exact = compute_analytical_solution(positions[:, 0], positions[:, 1])
        
        h = 1.0 / res
        l2_error = compute_l2_error(u_num, u_exact, h)
        errors.append(l2_error)
        
        print(f"  Resolution {res:3d}x{res:3d}: L2 error = {l2_error:.6e}")
    
    # Check that error decreases (roughly quadratically for linear elements)
    convergence_ok = all(errors[i] > errors[i+1] for i in range(len(errors)-1))
    
    # Check rough convergence rate (should be ~O(h²) for linear elements)
    if len(errors) >= 2:
        rate = np.log(errors[0] / errors[1]) / np.log(2)
        print(f"\n  Convergence rate (res 10→20): {rate:.2f} (expected ~2.0 for P1)")
    
    if convergence_ok:
        print("\n  PASSED: Error decreases with mesh refinement ✓")
    else:
        print("\n  FAILED: Error does not decrease properly")
    
    assert convergence_ok


def test_boundary_conditions():
    """Test that boundary conditions are satisfied."""
    print("\n" + "=" * 60)
    print("Test 2: Dirichlet boundary conditions")
    print("=" * 60)
    
    solver = PoissonSolver(resolution=20, degree=1)
    field = solver.solve(source_type="sin", bc_value=0.0)
    
    positions = solver.get_dof_positions()
    u_num = field.dof_values.numpy()
    
    # Find boundary nodes (x=0 or x=1 or y=0 or y=1)
    tol = 1e-10
    boundary_mask = (
        (np.abs(positions[:, 0]) < tol) | 
        (np.abs(positions[:, 0] - 1.0) < tol) |
        (np.abs(positions[:, 1]) < tol) | 
        (np.abs(positions[:, 1] - 1.0) < tol)
    )
    
    boundary_values = u_num[boundary_mask]
    max_boundary_error = np.max(np.abs(boundary_values))
    
    print(f"  Number of boundary nodes: {np.sum(boundary_mask)}")
    print(f"  Max boundary value (should be ~0): {max_boundary_error:.6e}")
    
    passed = max_boundary_error < 1e-6
    if passed:
        print("\n  PASSED: Boundary conditions satisfied ✓")
    else:
        print("\n  FAILED: Boundary conditions not satisfied")
    
    assert bool(passed)


def test_solver_consistency():
    """Test that solver produces consistent results across runs."""
    print("\n" + "=" * 60)
    print("Test 3: Solver consistency across runs")
    print("=" * 60)
    
    results = []
    for run in range(2):
        solver = PoissonSolver(resolution=15, degree=1)
        field = solver.solve(source_type="sin", bc_value=0.0)
        u_num = field.dof_values.numpy()
        results.append(u_num.copy())
        print(f"  Run {run + 1}: max value = {np.max(u_num):.10f}")
    
    diff = np.max(np.abs(results[0] - results[1]))
    print(f"\n  Max difference between runs: {diff:.6e}")
    
    passed = diff < 1e-10
    if passed:
        print("\n  PASSED: Solver produces consistent results ✓")
    else:
        print("\n  FAILED: Results differ between runs")
    
    assert bool(passed)


def test_known_solution():
    """Test against known analytical solution."""
    print("\n" + "=" * 60)
    print("Test 4: Comparison with analytical solution")
    print("=" * 60)
    
    solver = PoissonSolver(resolution=30, degree=1)
    field = solver.solve(source_type="sin", bc_value=0.0)
    
    positions = solver.get_dof_positions()
    u_num = field.dof_values.numpy()
    u_exact = compute_analytical_solution(positions[:, 0], positions[:, 1])
    
    # Check interior point (should be close to max value sin(π*0.5)*sin(π*0.5) = 1)
    center_idx = np.argmin(np.abs(positions[:, 0] - 0.5) + np.abs(positions[:, 1] - 0.5))
    center_num = u_num[center_idx]
    center_exact = 1.0  # sin(π/2) * sin(π/2) = 1
    
    print(f"  Center point (0.5, 0.5):")
    print(f"    Numerical:  {center_num:.6f}")
    print(f"    Analytical: {center_exact:.6f}")
    print(f"    Error:      {abs(center_num - center_exact):.6e}")
    
    # Overall L2 error
    h = 1.0 / 30
    l2_error = compute_l2_error(u_num, u_exact, h)
    print(f"\n  Overall L2 error: {l2_error:.6e}")
    
    # Expect small error for fine mesh
    passed = l2_error < 1e-4
    if passed:
        print("\n  PASSED: Solution matches analytical result ✓")
    else:
        print("\n  FAILED: L2 error too large")
    
    assert bool(passed)


if __name__ == "__main__":
    print("=" * 60)
    print("Poisson Solver Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Convergence", test_convergence_with_resolution),
        ("Boundary Conditions", test_boundary_conditions),
        ("Consistency", test_solver_consistency),
        ("Analytical Solution", test_known_solution),
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    sys.exit(0 if passed == len(tests) else 1)
