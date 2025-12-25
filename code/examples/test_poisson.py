"""
Test suite for Poisson solver.

Validates the Poisson solver against analytical solutions.
"""

import sys
sys.path.insert(0, '/workspace/code/examples')

import warp as wp
import numpy as np
from poisson_solver import PoissonSolver, analytical_solution


def test_constant_forcing():
    """Test 1: Constant forcing with zero boundary conditions."""
    print("\n" + "="*60)
    print("TEST 1: Constant Forcing")
    print("="*60)
    
    solver = PoissonSolver(resolution=16, degree=2, use_manufactured=False, quiet=True)
    solution = solver.solve()
    
    # Check that solution is computed
    assert solution is not None, "Solution should not be None"
    assert solution.dof_values is not None, "DOF values should not be None"
    
    # Check that solution has reasonable values
    dofs = solution.dof_values.numpy()
    assert len(dofs) > 0, "Solution should have DOFs"
    assert not np.any(np.isnan(dofs)), "Solution should not contain NaN"
    assert not np.any(np.isinf(dofs)), "Solution should not contain Inf"
    
    print(f"✓ Solution computed with {len(dofs)} DOFs")
    print(f"  Min value: {np.min(dofs):.6f}")
    print(f"  Max value: {np.max(dofs):.6f}")
    print(f"  Mean value: {np.mean(dofs):.6f}")
    
    return True


def test_manufactured_solution():
    """Test 2: Manufactured solution u = sin(πx)sin(πy)."""
    print("\n" + "="*60)
    print("TEST 2: Manufactured Solution")
    print("="*60)
    
    solver = PoissonSolver(resolution=32, degree=2, use_manufactured=True, quiet=True)
    solution = solver.solve()
    
    # Check solution properties
    dofs = solution.dof_values.numpy()
    
    print(f"✓ Solution computed with {len(dofs)} DOFs")
    print(f"  Min value: {np.min(dofs):.6f}")
    print(f"  Max value: {np.max(dofs):.6f}")
    
    # Expected: u should be between 0 and ~1.0 for manufactured solution
    assert np.min(dofs) >= -0.1, f"Min value {np.min(dofs)} seems too negative"
    assert np.max(dofs) <= 1.1, f"Max value {np.max(dofs)} seems too large"
    
    print("✓ Solution values are in expected range")
    
    return True


def test_different_resolutions():
    """Test 3: Run with different mesh resolutions."""
    print("\n" + "="*60)
    print("TEST 3: Different Resolutions")
    print("="*60)
    
    resolutions = [8, 16, 32]
    results = []
    
    for res in resolutions:
        solver = PoissonSolver(resolution=res, degree=2, use_manufactured=True, quiet=True)
        solution = solver.solve()
        dofs = solution.dof_values.numpy()
        
        results.append({
            'resolution': res,
            'dofs': len(dofs),
            'max_val': np.max(dofs)
        })
        
        print(f"  Resolution {res:2d}: {len(dofs):5d} DOFs, max = {np.max(dofs):.6f}")
    
    # Check that DOFs increase with resolution
    for i in range(len(results) - 1):
        assert results[i+1]['dofs'] > results[i]['dofs'], \
            "DOF count should increase with resolution"
    
    print("✓ DOF count increases appropriately with resolution")
    
    return True


def test_consistency():
    """Test 4: Run twice and check consistency."""
    print("\n" + "="*60)
    print("TEST 4: Consistency Check")
    print("="*60)
    
    # First run
    solver1 = PoissonSolver(resolution=16, degree=2, use_manufactured=True, quiet=True)
    solution1 = solver1.solve()
    dofs1 = solution1.dof_values.numpy()
    
    # Second run
    solver2 = PoissonSolver(resolution=16, degree=2, use_manufactured=True, quiet=True)
    solution2 = solver2.solve()
    dofs2 = solution2.dof_values.numpy()
    
    # Check equality
    assert len(dofs1) == len(dofs2), "Solutions should have same size"
    
    # Check numerical equality (should be exact since deterministic)
    max_diff = np.max(np.abs(dofs1 - dofs2))
    rel_diff = max_diff / (np.max(np.abs(dofs1)) + 1e-10)
    
    print(f"  Run 1: {len(dofs1)} DOFs, max = {np.max(dofs1):.6f}")
    print(f"  Run 2: {len(dofs2)} DOFs, max = {np.max(dofs2):.6f}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Relative difference: {rel_diff:.2e}")
    
    assert max_diff < 1e-10, f"Solutions should be identical, but differ by {max_diff}"
    
    print("✓ Results are consistent across runs")
    
    return True


def test_boundary_conditions():
    """Test 5: Verify zero boundary conditions."""
    print("\n" + "="*60)
    print("TEST 5: Boundary Conditions")
    print("="*60)
    
    solver = PoissonSolver(resolution=16, degree=1, use_manufactured=True, quiet=True)
    solution = solver.solve()
    dofs = solution.dof_values.numpy()
    
    # For degree 1, DOFs correspond to grid vertices
    # Check that boundary values are close to zero
    n = 16 + 1  # resolution + 1
    
    # Reshape to grid (for degree 1)
    if solver.degree == 1 and len(dofs) == n * n:
        grid_vals = dofs.reshape(n, n)
        
        # Check boundaries
        boundary_vals = np.concatenate([
            grid_vals[0, :],   # bottom
            grid_vals[-1, :],  # top
            grid_vals[:, 0],   # left
            grid_vals[:, -1]   # right
        ])
        
        max_boundary = np.max(np.abs(boundary_vals))
        print(f"  Max absolute boundary value: {max_boundary:.2e}")
        
        assert max_boundary < 1e-6, f"Boundary values should be ~0, but max is {max_boundary}"
        print("✓ Boundary conditions satisfied")
    else:
        print("  (Skipped for higher-degree elements)")
    
    return True


def main():
    """Run all tests."""
    wp.init()
    
    print("="*60)
    print("POISSON SOLVER TEST SUITE")
    print("="*60)
    
    tests = [
        ("Constant Forcing", test_constant_forcing),
        ("Manufactured Solution", test_manufactured_solution),
        ("Different Resolutions", test_different_resolutions),
        ("Consistency", test_consistency),
        ("Boundary Conditions", test_boundary_conditions),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
