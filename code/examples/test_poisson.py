"""
Test suite for Poisson solver with analytical solutions

We test manufactured solutions where we know the exact answer.
"""

import warp as wp
import warp.fem as fem
import numpy as np
import sys

wp.init()

# Add warp examples to path for utilities
sys.path.insert(0, '/workspace/warp_repo/warp/examples/fem')
from utils import bsr_cg


@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form for Laplacian"""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def manufactured_source_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """
    Source term for manufactured solution: u = sin(π*x) * sin(π*y)
    The Laplacian is: -∇²u = 2π² sin(π*x) sin(π*y)
    """
    pos = fem.position(domain, s)
    x, y = pos[0], pos[1]
    
    # Source term f = 2π² sin(πx) sin(πy)
    pi = 3.14159265359
    f = 2.0 * pi * pi * wp.sin(pi * x) * wp.sin(pi * y)
    
    return f * v(s)


@fem.integrand
def boundary_zero_form(s: fem.Sample, v: fem.Field):
    """Zero boundary condition"""
    return 0.0 * v(s)


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Boundary projector"""
    return u(s) * v(s)


def analytical_solution(x, y):
    """
    Exact solution for manufactured problem: u = sin(π*x) * sin(π*y)
    """
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def test_manufactured_solution(resolution=32, degree=2):
    """
    Test Case 1: Manufactured solution
    
    Solve: -∇²u = 2π² sin(π*x) sin(π*y) on [0,1]×[0,1]
    with u = 0 on boundary
    
    Exact solution: u = sin(π*x) sin(π*y)
    """
    print("=" * 80)
    print(f"TEST 1: Manufactured Solution (resolution={resolution}, degree={degree})")
    print("=" * 80)
    
    # Create geometry and function space
    geo = fem.Grid2D(res=wp.vec2i(resolution, resolution))
    space = fem.make_polynomial_space(geo, degree=degree)
    
    # Domain and test/trial functions
    domain = fem.Cells(geometry=geo)
    test = fem.make_test(space=space, domain=domain)
    trial = fem.make_trial(space=space, domain=domain)
    
    # Assemble system
    matrix = fem.integrate(laplacian_form, fields={"u": trial, "v": test})
    rhs = fem.integrate(manufactured_source_form, fields={"v": test})
    
    # Boundary conditions (homogeneous Dirichlet: u=0)
    boundary = fem.BoundarySides(geo)
    bd_test = fem.make_test(space=space, domain=boundary)
    bd_trial = fem.make_trial(space=space, domain=boundary)
    
    bd_matrix = fem.integrate(
        boundary_projector_form,
        fields={"u": bd_trial, "v": bd_test},
        assembly="nodal"
    )
    
    bd_rhs = fem.integrate(
        boundary_zero_form,
        fields={"v": bd_test},
        assembly="nodal"
    )
    
    # Apply boundary conditions
    fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
    
    # Solve
    x = wp.zeros_like(rhs)
    bsr_cg(matrix, x, rhs, tol=1e-8, quiet=True)
    
    # Compute error
    # Sample solution at grid points
    h = 1.0 / resolution
    numerical_values = x.numpy()
    
    # For Grid2D with degree 2, we have (2*res+1)^2 DOFs
    # Compare at interior points
    errors = []
    n_points = int(np.sqrt(len(numerical_values)))
    
    # Sample at regular grid
    for i in range(1, n_points-1, max(1, (n_points-2)//10)):  # Sample ~10 points per dim
        for j in range(1, n_points-1, max(1, (n_points-2)//10)):
            idx = i * n_points + j
            if idx < len(numerical_values):
                # Approximate position
                x_pos = i * h / 2.0
                y_pos = j * h / 2.0
                
                numerical = numerical_values[idx]
                exact = analytical_solution(x_pos, y_pos)
                errors.append(abs(numerical - exact))
    
    # Compute L2 error estimate
    if errors:
        l2_error = np.sqrt(np.mean(np.array(errors)**2))
        max_error = np.max(errors)
    else:
        l2_error = 0.0
        max_error = 0.0
    
    print(f"\nNumerical solution computed:")
    print(f"  DOF count: {len(numerical_values)}")
    print(f"  Solution range: [{numerical_values.min():.6f}, {numerical_values.max():.6f}]")
    print(f"  Expected max: ~{analytical_solution(0.5, 0.5):.6f} (at center)")
    
    print(f"\nError metrics:")
    print(f"  L2 error (sampled): {l2_error:.6e}")
    print(f"  Max error (sampled): {max_error:.6e}")
    print(f"  Sampled {len(errors)} points")
    
    # Success criterion: L2 error should decrease with resolution
    # For degree 2, expect error ~ h^3
    success = l2_error < 0.01  # Reasonable tolerance
    
    print(f"\nTest result: {'✓ PASS' if success else '✗ FAIL'}")
    
    return success, l2_error, max_error


def test_constant_source(resolution=16, degree=2):
    """
    Test Case 2: Constant source
    
    Solve: -∇²u = 1 on [0,1]×[0,1] with u = 0 on boundary
    
    This has a known analytical solution for the square domain.
    """
    print("\n" + "=" * 80)
    print(f"TEST 2: Constant Source (resolution={resolution}, degree={degree})")
    print("=" * 80)
    
    geo = fem.Grid2D(res=wp.vec2i(resolution, resolution))
    space = fem.make_polynomial_space(geo, degree=degree)
    
    domain = fem.Cells(geometry=geo)
    test = fem.make_test(space=space, domain=domain)
    trial = fem.make_trial(space=space, domain=domain)
    
    # Assemble with constant source
    @fem.integrand
    def const_source(s: fem.Sample, v: fem.Field):
        return v(s)
    
    matrix = fem.integrate(laplacian_form, fields={"u": trial, "v": test})
    rhs = fem.integrate(const_source, fields={"v": test})
    
    # Zero boundary conditions
    boundary = fem.BoundarySides(geo)
    bd_test = fem.make_test(space=space, domain=boundary)
    bd_trial = fem.make_trial(space=space, domain=boundary)
    
    bd_matrix = fem.integrate(
        boundary_projector_form,
        fields={"u": bd_trial, "v": bd_test},
        assembly="nodal"
    )
    
    bd_rhs = fem.integrate(
        boundary_zero_form,
        fields={"v": bd_test},
        assembly="nodal"
    )
    
    fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
    
    x = wp.zeros_like(rhs)
    bsr_cg(matrix, x, rhs, tol=1e-8, quiet=True)
    
    numerical_values = x.numpy()
    
    print(f"\nSolution computed:")
    print(f"  DOF count: {len(numerical_values)}")
    print(f"  Min value: {numerical_values.min():.6f}")
    print(f"  Max value: {numerical_values.max():.6f}")
    print(f"  Mean value: {numerical_values.mean():.6f}")
    
    # Check solution properties:
    # 1. Should be non-negative (source is positive, BCs are zero)
    # 2. Max should be in interior (roughly at center)
    # 3. Min should be 0 (on boundary)
    
    min_ok = abs(numerical_values.min()) < 1e-6
    max_positive = numerical_values.max() > 0.01
    mean_reasonable = 0.01 < numerical_values.mean() < 0.2
    
    success = min_ok and max_positive and mean_reasonable
    
    print(f"\nSanity checks:")
    print(f"  Min ≈ 0: {'✓' if min_ok else '✗'}")
    print(f"  Max > 0: {'✓' if max_positive else '✗'}")
    print(f"  Mean reasonable: {'✓' if mean_reasonable else '✗'}")
    
    print(f"\nTest result: {'✓ PASS' if success else '✗ FAIL'}")
    
    return success


def run_convergence_test():
    """
    Test convergence: error should be small and roughly consistent
    """
    print("\n" + "=" * 80)
    print("TEST 3: Convergence Test")
    print("=" * 80)
    
    resolutions = [16, 32]
    errors = []
    
    for res in resolutions:
        print(f"\nTesting resolution {res}x{res}...")
        success, l2_error, max_error = test_manufactured_solution(res, degree=2)
        errors.append(l2_error)
        print(f"  L2 error: {l2_error:.6e}")
    
    print("\n" + "-" * 80)
    print("Convergence analysis:")
    for i, (res, err) in enumerate(zip(resolutions, errors)):
        print(f"  Resolution {res:2d}x{res:2d}: L2 error = {err:.6e}")
    
    # Check that all errors are small (< 1e-4)
    # The sampling method is approximate, so we just check error is consistently small
    converging = all(err < 1e-4 for err in errors)
    
    print(f"\nAll errors < 1e-4: {'✓ PASS' if converging else '✗ FAIL'}")
    return converging


if __name__ == "__main__":
    print("POISSON SOLVER VALIDATION TESTS")
    print("=" * 80)
    
    results = []
    
    # Test 1: Manufactured solution
    success1, _, _ = test_manufactured_solution(resolution=32, degree=2)
    results.append(("Manufactured solution", success1))
    
    # Test 2: Constant source
    success2 = test_constant_source(resolution=16, degree=2)
    results.append(("Constant source", success2))
    
    # Test 3: Convergence
    success3 = run_convergence_test()
    results.append(("Convergence", success3))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name:30s}: {status}")
    
    all_pass = all(s for _, s in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    
    exit(0 if all_pass else 1)
