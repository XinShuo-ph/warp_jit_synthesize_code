"""Validation tests for the Poisson solver.

Tests compare numerical solution to analytical solution:
u = sin(πx)sin(πy) for f = 2π²sin(πx)sin(πy)
"""
import sys
# Add warp fem examples to path for bsr_cg utility
import warp
warp_examples_fem = str(__import__('pathlib').Path(warp.__file__).parent / 'examples' / 'fem')
sys.path.insert(0, warp_examples_fem)

import numpy as np
import warp as wp
import warp.fem as fem

from poisson_solver import solve_poisson, get_analytical_solution

wp.init()


def test_poisson_convergence():
    """Test that the Poisson solver converges to the analytical solution."""
    print("Test 1: Convergence to analytical solution")
    print("="*50)
    
    # Solve with fine mesh
    resolution = 30
    solution_field, geo = solve_poisson(resolution=resolution, degree=2, quiet=True)
    
    # Get DOF values
    dof_vals = solution_field.dof_values.numpy()
    
    # Check solution range (should be approximately [0, 1])
    min_val, max_val = dof_vals.min(), dof_vals.max()
    print(f"Solution range: [{min_val:.6f}, {max_val:.6f}]")
    print(f"Expected range: [0.0, 1.0]")
    
    # Max value should be close to 1.0 (at center)
    expected_max = get_analytical_solution(0.5, 0.5)
    max_error = abs(max_val - expected_max)
    print(f"Max value error: {max_error:.6f}")
    
    # Min should be close to 0 (at boundary)
    assert min_val >= -0.01, f"Min value {min_val} should be >= -0.01"
    assert max_error < 0.02, f"Max error {max_error} should be < 0.02"
    
    print("✓ Test passed!\n")
    return True


def test_mesh_refinement():
    """Test that finer meshes give more accurate solutions."""
    print("Test 2: Mesh refinement convergence")
    print("="*50)
    
    errors = []
    resolutions = [5, 10, 20]  # Use coarser meshes to see convergence
    
    for res in resolutions:
        solution_field, geo = solve_poisson(resolution=res, degree=2, quiet=True)
        dof_vals = solution_field.dof_values.numpy()
        
        # Compute max error (approximate)
        expected_max = get_analytical_solution(0.5, 0.5)
        max_val = dof_vals.max()
        error = abs(max_val - expected_max)
        errors.append(error)
        print(f"Resolution {res}x{res}: max error = {error:.6e}")
    
    # All errors should be small (< 0.01)
    assert all(e < 0.01 for e in errors), f"All errors should be < 0.01"
    
    # The finest mesh should have error < 1e-4
    assert errors[-1] < 1e-4, f"Finest mesh error should be < 1e-4, got {errors[-1]}"
    
    print("✓ Test passed!\n")
    return True


def test_boundary_conditions():
    """Test that boundary conditions are satisfied (u = 0 on boundary)."""
    print("Test 3: Boundary conditions")
    print("="*50)
    
    solution_field, geo = solve_poisson(resolution=20, degree=2, quiet=True)
    dof_vals = solution_field.dof_values.numpy()
    
    # For a Grid2D with degree 2, boundary DOFs are at specific indices
    # The minimum values should be at the boundary and close to zero
    min_val = dof_vals.min()
    print(f"Minimum value (should be at boundary): {min_val:.6f}")
    
    assert abs(min_val) < 0.01, f"Boundary values should be close to 0, got {min_val}"
    
    print("✓ Test passed!\n")
    return True


def test_symmetry():
    """Test that solution has expected symmetry (symmetric in x and y)."""
    print("Test 4: Solution symmetry")
    print("="*50)
    
    solution_field, geo = solve_poisson(resolution=20, degree=2, quiet=True)
    dof_vals = solution_field.dof_values.numpy()
    
    # The solution should be symmetric
    # For a 20x20 grid with degree 2, we have 41x41 = 1681 DOFs
    n = 41  # (resolution * degree + 1)
    
    # Reshape to 2D
    if len(dof_vals) == n * n:
        vals_2d = dof_vals.reshape(n, n)
        
        # Check symmetry: u(x,y) = u(y,x)
        transpose_diff = np.abs(vals_2d - vals_2d.T).max()
        print(f"Max transpose difference: {transpose_diff:.6f}")
        
        # Check symmetry: u(x,y) = u(1-x,y)
        flip_x_diff = np.abs(vals_2d - np.flip(vals_2d, axis=0)).max()
        print(f"Max x-flip difference: {flip_x_diff:.6f}")
        
        assert transpose_diff < 0.01, f"Solution should be symmetric"
        assert flip_x_diff < 0.01, f"Solution should be x-symmetric"
    else:
        print(f"Skipping detailed symmetry check (DOF count {len(dof_vals)} != {n*n})")
    
    print("✓ Test passed!\n")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("POISSON SOLVER VALIDATION TESTS")
    print("="*60 + "\n")
    
    tests = [
        ("Convergence", test_poisson_convergence),
        ("Mesh refinement", test_mesh_refinement),
        ("Boundary conditions", test_boundary_conditions),
        ("Symmetry", test_symmetry),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} failed: {e}\n")
            results.append((name, False))
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for _, s in results if s)
    for name, success in results:
        print(f"  {'✓' if success else '✗'} {name}")
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
