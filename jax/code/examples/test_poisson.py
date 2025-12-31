"""Test suite for Poisson solver with analytical solutions."""
import jax.numpy as jnp
from poisson_solver import solve_poisson, create_grid, laplacian_2d


def test_sin_sin():
    """Test case 1: u = sin(πx)sin(πy)
    
    Analytical: ∇²u = -2π²sin(πx)sin(πy)
    Boundary: u = 0 on all edges (Dirichlet)
    """
    print("=" * 60)
    print("Test 1: u = sin(πx)sin(πy)")
    print("=" * 60)
    
    nx, ny = 64, 64
    X, Y, dx, dy = create_grid(nx, ny)
    
    # Analytical solution
    u_exact = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    
    # Source term: f = ∇²u = -2π²sin(πx)sin(πy)
    f = -2 * jnp.pi**2 * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    
    # Boundary conditions (all zero for this case)
    u0 = jnp.zeros((ny, nx))
    
    # Solve
    u_num, converged, n_iters = solve_poisson(f, u0, dx, dy, tol=1e-6, max_iters=10000)
    
    # Compute errors
    max_error = float(jnp.max(jnp.abs(u_num - u_exact)))
    l2_error = float(jnp.sqrt(jnp.mean((u_num - u_exact)**2)))
    
    print(f"  Converged: {converged}")
    print(f"  Iterations: {n_iters}")
    print(f"  Max error: {max_error:.6e}")
    print(f"  L2 error: {l2_error:.6e}")
    
    passed = converged and max_error < 1e-3
    print(f"  PASS: {passed}")
    
    return passed, max_error


def test_quadratic():
    """Test case 2: u = x² + y²
    
    Analytical: ∇²u = 2 + 2 = 4
    Boundary: u = x² + y² on edges (non-zero Dirichlet)
    """
    print("=" * 60)
    print("Test 2: u = x² + y²")
    print("=" * 60)
    
    nx, ny = 64, 64
    X, Y, dx, dy = create_grid(nx, ny)
    
    # Analytical solution
    u_exact = X**2 + Y**2
    
    # Source term: f = ∇²u = 4
    f = jnp.ones((ny, nx)) * 4.0
    
    # Initial guess with boundary conditions
    u0 = jnp.zeros((ny, nx))
    
    # Set boundary conditions
    u0 = u0.at[0, :].set(X[0, :]**2 + Y[0, :]**2)    # bottom
    u0 = u0.at[-1, :].set(X[-1, :]**2 + Y[-1, :]**2)  # top
    u0 = u0.at[:, 0].set(X[:, 0]**2 + Y[:, 0]**2)    # left
    u0 = u0.at[:, -1].set(X[:, -1]**2 + Y[:, -1]**2)  # right
    
    # Solve
    u_num, converged, n_iters = solve_poisson(f, u0, dx, dy, tol=1e-6, max_iters=10000)
    
    # Compute errors (interior only, since boundaries are exact)
    interior = (slice(1, -1), slice(1, -1))
    max_error = float(jnp.max(jnp.abs(u_num[interior] - u_exact[interior])))
    l2_error = float(jnp.sqrt(jnp.mean((u_num[interior] - u_exact[interior])**2)))
    
    print(f"  Converged: {converged}")
    print(f"  Iterations: {n_iters}")
    print(f"  Max error (interior): {max_error:.6e}")
    print(f"  L2 error (interior): {l2_error:.6e}")
    
    passed = converged and max_error < 1e-3
    print(f"  PASS: {passed}")
    
    return passed, max_error


def test_laplacian():
    """Verify Laplacian operator computes correct values."""
    print("=" * 60)
    print("Test 3: Laplacian operator verification")
    print("=" * 60)
    
    nx, ny = 32, 32
    X, Y, dx, dy = create_grid(nx, ny)
    
    # Test with u = sin(πx)sin(πy)
    u = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    lap_exact = -2 * jnp.pi**2 * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    
    lap_num = laplacian_2d(u, dx, dy)
    
    # Check interior points
    interior = (slice(2, -2), slice(2, -2))
    max_error = float(jnp.max(jnp.abs(lap_num[interior] - lap_exact[interior])))
    
    print(f"  Max error: {max_error:.6e}")
    
    # Finite difference has O(h²) error, with h≈0.03, expect error ~ 0.02
    passed = max_error < 0.1
    print(f"  PASS: {passed}")
    
    return passed, max_error


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "=" * 60)
    print("POISSON SOLVER TEST SUITE")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run all tests
    results.append(("Laplacian operator", *test_laplacian()))
    print()
    results.append(("sin(πx)sin(πy)", *test_sin_sin()))
    print()
    results.append(("x² + y²", *test_quadratic()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed, error in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status} (error: {error:.2e})")
        all_passed = all_passed and passed
    
    print(f"\nAll tests passed: {all_passed}")
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
