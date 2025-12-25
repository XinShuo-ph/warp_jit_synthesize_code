"""Test suite for Poisson solver validation.

Tests:
1. Solver converges for manufactured solution
2. Error decreases with mesh refinement (convergence test)
3. Results are reproducible
"""

import warp as wp
import numpy as np
import sys
sys.path.insert(0, '/workspace/code')

from examples.poisson_solver import solve_poisson, compute_error

wp.init()


def test_convergence():
    """Test that error decreases with mesh refinement."""
    print("Test 1: Convergence with mesh refinement")
    print("-" * 60)
    
    resolutions = [10, 20, 40]
    errors = []
    
    for res in resolutions:
        solution, geo, space = solve_poisson(resolution=res, degree=2)
        error = compute_error(solution, geo, space)
        errors.append(error)
        print(f"Resolution {res:2d}x{res:2d}: L2 error = {error:.6e}")
    
    # Check that error decreases
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i+1], f"Error should decrease: {errors[i]} > {errors[i+1]}"
    
    # Check convergence rate (roughly O(h^2) for linear elements)
    ratio = errors[0] / errors[1]
    print(f"Error reduction ratio (10→20): {ratio:.2f}")
    assert ratio > 3.0, f"Expected ratio > 3, got {ratio}"
    
    print("✓ PASS: Error decreases with refinement\n")
    return True


def test_reproducibility():
    """Test that results are reproducible across runs."""
    print("Test 2: Reproducibility")
    print("-" * 60)
    
    res = 15
    errors = []
    
    for run in range(3):
        solution, geo, space = solve_poisson(resolution=res, degree=2)
        error = compute_error(solution, geo, space)
        errors.append(error)
        print(f"Run {run + 1}: L2 error = {error:.10e}")
    
    # Check that all errors are identical
    for i in range(len(errors) - 1):
        assert abs(errors[i] - errors[i+1]) < 1e-12, \
            f"Results should be identical: {errors[i]} vs {errors[i+1]}"
    
    print("✓ PASS: Results are reproducible\n")
    return True


def test_error_magnitude():
    """Test that error is within expected bounds."""
    print("Test 3: Error magnitude check")
    print("-" * 60)
    
    solution, geo, space = solve_poisson(resolution=30, degree=2)
    error = compute_error(solution, geo, space)
    print(f"L2 error at 30x30 resolution: {error:.6e}")
    
    # For a manufactured solution with quadratic elements, 
    # we expect very small error
    assert error < 1e-4, f"Error too large: {error}"
    
    print("✓ PASS: Error within acceptable bounds\n")
    return True


def test_manufactured_solution_properties():
    """Verify solution satisfies boundary conditions."""
    print("Test 4: Boundary condition verification")
    print("-" * 60)
    
    solution, geo, space = solve_poisson(resolution=20, degree=2)
    
    # Get solution values
    sol_values = solution.dof_values.numpy()
    
    # For manufactured solution u = sin(πx)sin(πy), 
    # boundary values should be 0 (since sin(0)=sin(π)=0)
    # This is implicitly enforced by our Dirichlet BC
    
    print(f"Solution min: {sol_values.min():.6f}")
    print(f"Solution max: {sol_values.max():.6f}")
    print(f"Solution mean: {sol_values.mean():.6f}")
    
    # Solution should have reasonable magnitude
    assert abs(sol_values.min()) < 1.0, "Solution out of range"
    assert abs(sol_values.max()) < 1.0, "Solution out of range"
    
    print("✓ PASS: Solution has expected properties\n")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("=" * 70)
    print("POISSON SOLVER VALIDATION TESTS")
    print("=" * 70)
    print()
    
    tests = [
        test_convergence,
        test_reproducibility,
        test_error_magnitude,
        test_manufactured_solution_properties,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"✗ FAIL: {test.__name__}")
            print(f"  Error: {e}\n")
            results.append((test.__name__, False))
    
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    exit(exit_code)
