#!/usr/bin/env python3
"""
Test suite for Poisson equation solvers

Tests numerical solutions against analytical solutions for various
forcing functions and boundary conditions.
"""
import jax.numpy as jnp
from poisson_solver import (
    PoissonSolver1D, PoissonSolver2D,
    forcing_sin_1d, analytical_sin_1d,
    forcing_poly_1d, analytical_poly_1d,
    forcing_sin_2d, analytical_sin_2d
)


class TestPoisson1D:
    """Tests for 1D Poisson solver"""
    
    def test_sin_forcing_coarse(self):
        """Test with sin forcing on coarse grid"""
        solver = PoissonSolver1D(n_points=51, domain=(0.0, 1.0))
        u_numerical = solver.solve(forcing_sin_1d, bc_left=0.0, bc_right=0.0)
        u_analytical = analytical_sin_1d(solver.x)
        
        l2_error = jnp.sqrt(jnp.mean((u_numerical - u_analytical)**2))
        assert l2_error < 5e-4, f"L2 error too large: {l2_error}"
    
    def test_sin_forcing_fine(self):
        """Test with sin forcing on fine grid"""
        solver = PoissonSolver1D(n_points=101, domain=(0.0, 1.0))
        u_numerical = solver.solve(forcing_sin_1d, bc_left=0.0, bc_right=0.0)
        u_analytical = analytical_sin_1d(solver.x)
        
        l2_error = jnp.sqrt(jnp.mean((u_numerical - u_analytical)**2))
        max_error = jnp.max(jnp.abs(u_numerical - u_analytical))
        
        print(f"\n   Fine grid L2 error: {l2_error:.6e}")
        print(f"   Fine grid max error: {max_error:.6e}")
        
        assert l2_error < 1e-4, f"L2 error too large: {l2_error}"
        assert max_error < 2e-4, f"Max error too large: {max_error}"
    
    def test_poly_forcing(self):
        """Test with polynomial forcing: f(x) = -2, u(x) = x(1-x)"""
        solver = PoissonSolver1D(n_points=51, domain=(0.0, 1.0))
        u_numerical = solver.solve(forcing_poly_1d, bc_left=0.0, bc_right=0.0)
        u_analytical = analytical_poly_1d(solver.x)
        
        l2_error = jnp.sqrt(jnp.mean((u_numerical - u_analytical)**2))
        
        print(f"\n   Polynomial forcing L2 error: {l2_error:.6e}")
        
        assert l2_error < 1e-4, f"L2 error too large: {l2_error}"
    
    def test_boundary_conditions(self):
        """Test with non-zero boundary conditions"""
        solver = PoissonSolver1D(n_points=51, domain=(0.0, 1.0))
        
        # Simple case: f=0, u(0)=1, u(1)=2 should give linear solution
        def forcing_zero(x):
            return jnp.zeros_like(x)
        
        u = solver.solve(forcing_zero, bc_left=1.0, bc_right=2.0)
        
        # Analytical solution for f=0: u(x) = 1 + x
        u_analytical = 1.0 + solver.x
        
        l2_error = jnp.sqrt(jnp.mean((u - u_analytical)**2))
        assert l2_error < 1e-4, f"BC test failed: {l2_error}"
    
    def test_convergence(self):
        """Test that error decreases with grid refinement"""
        errors = []
        ns = [21, 41, 81]
        
        for n in ns:
            solver = PoissonSolver1D(n_points=n, domain=(0.0, 1.0))
            u_numerical = solver.solve(forcing_sin_1d, bc_left=0.0, bc_right=0.0)
            u_analytical = analytical_sin_1d(solver.x)
            l2_error = jnp.sqrt(jnp.mean((u_numerical - u_analytical)**2))
            errors.append(l2_error)
        
        print(f"\n   Convergence test:")
        for n, err in zip(ns, errors):
            print(f"   n={n:3d}: L2 error = {err:.6e}")
        
        # Error should decrease with refinement
        assert errors[1] < errors[0], "Error didn't decrease from n=21 to n=41"
        assert errors[2] < errors[1], "Error didn't decrease from n=41 to n=81"


class TestPoisson2D:
    """Tests for 2D Poisson solver"""
    
    def test_sin_forcing_2d(self):
        """Test 2D solver with sin(πx)sin(πy) forcing"""
        solver = PoissonSolver2D(nx=41, ny=41, domain=((0.0, 1.0), (0.0, 1.0)))
        u_numerical = solver.solve_jacobi(forcing_sin_2d, n_iter=2000, bc_value=0.0)
        u_analytical = analytical_sin_2d(solver.X, solver.Y)
        
        l2_error = jnp.sqrt(jnp.mean((u_numerical - u_analytical)**2))
        max_error = jnp.max(jnp.abs(u_numerical - u_analytical))
        
        print(f"\n   2D L2 error: {l2_error:.6e}")
        print(f"   2D max error: {max_error:.6e}")
        
        # Jacobi is slower to converge, so we allow larger error
        assert l2_error < 2e-2, f"L2 error too large: {l2_error}"
        assert max_error < 4e-2, f"Max error too large: {max_error}"
    
    def test_boundary_values(self):
        """Test that boundary conditions are enforced"""
        solver = PoissonSolver2D(nx=21, ny=21, domain=((0.0, 1.0), (0.0, 1.0)))
        u = solver.solve_jacobi(forcing_sin_2d, n_iter=100, bc_value=0.0)
        
        # Check all boundaries are zero
        assert jnp.allclose(u[0, :], 0.0), "Left boundary not zero"
        assert jnp.allclose(u[-1, :], 0.0), "Right boundary not zero"
        assert jnp.allclose(u[:, 0], 0.0), "Bottom boundary not zero"
        assert jnp.allclose(u[:, -1], 0.0), "Top boundary not zero"
    
    def test_symmetry(self):
        """Test symmetry of solution for symmetric forcing"""
        solver = PoissonSolver2D(nx=41, ny=41, domain=((0.0, 1.0), (0.0, 1.0)))
        u = solver.solve_jacobi(forcing_sin_2d, n_iter=1000, bc_value=0.0)
        
        # Solution should be symmetric: u(x,y) = u(y,x) for this forcing
        max_asymmetry = jnp.max(jnp.abs(u - u.T))
        
        print(f"\n   Max asymmetry: {max_asymmetry:.6e}")
        
        assert max_asymmetry < 1e-6, f"Solution not symmetric: {max_asymmetry}"


def test_reproducibility_1d():
    """Test that running twice gives same results (deterministic)"""
    solver = PoissonSolver1D(n_points=51, domain=(0.0, 1.0))
    
    u1 = solver.solve(forcing_sin_1d, bc_left=0.0, bc_right=0.0)
    u2 = solver.solve(forcing_sin_1d, bc_left=0.0, bc_right=0.0)
    
    assert jnp.allclose(u1, u2), "1D solver not deterministic"


def test_reproducibility_2d():
    """Test that running twice gives same results (deterministic)"""
    solver = PoissonSolver2D(nx=31, ny=31, domain=((0.0, 1.0), (0.0, 1.0)))
    
    u1 = solver.solve_jacobi(forcing_sin_2d, n_iter=500, bc_value=0.0)
    u2 = solver.solve_jacobi(forcing_sin_2d, n_iter=500, bc_value=0.0)
    
    assert jnp.allclose(u1, u2), "2D solver not deterministic"


if __name__ == "__main__":
    print("=" * 70)
    print("Running Poisson Solver Tests")
    print("=" * 70)
    
    # Run tests manually (can also use pytest)
    test_1d = TestPoisson1D()
    test_2d = TestPoisson2D()
    
    print("\n1D TESTS")
    print("-" * 70)
    
    print("Test 1: Sin forcing (coarse)")
    test_1d.test_sin_forcing_coarse()
    print("   ✓ Passed")
    
    print("Test 2: Sin forcing (fine)")
    test_1d.test_sin_forcing_fine()
    print("   ✓ Passed")
    
    print("Test 3: Polynomial forcing")
    test_1d.test_poly_forcing()
    print("   ✓ Passed")
    
    print("Test 4: Boundary conditions")
    test_1d.test_boundary_conditions()
    print("   ✓ Passed")
    
    print("Test 5: Convergence")
    test_1d.test_convergence()
    print("   ✓ Passed")
    
    print("\n2D TESTS")
    print("-" * 70)
    
    print("Test 6: Sin forcing 2D")
    test_2d.test_sin_forcing_2d()
    print("   ✓ Passed")
    
    print("Test 7: Boundary values")
    test_2d.test_boundary_values()
    print("   ✓ Passed")
    
    print("Test 8: Symmetry")
    test_2d.test_symmetry()
    print("   ✓ Passed")
    
    print("\nREPRODUCIBILITY TESTS")
    print("-" * 70)
    
    print("Test 9: Reproducibility 1D")
    test_reproducibility_1d()
    print("   ✓ Passed")
    
    print("Test 10: Reproducibility 2D")
    test_reproducibility_2d()
    print("   ✓ Passed")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
