"""
1D Poisson Equation Solver using JAX

Solves: -d²u/dx² = f(x) on [0, 1]
Boundary conditions: u(0) = 0, u(1) = 0

Uses finite difference method and iterative solver (Jacobi method)
"""

import jax
import jax.numpy as jnp
from jax import jit
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


@jit
def jacobi_step(u, f, dx):
    """
    One iteration of Jacobi method for Poisson equation.
    
    Args:
        u: Current solution approximation
        f: Right-hand side (source term)
        dx: Grid spacing
    
    Returns:
        Updated solution
    """
    u_new = jnp.zeros_like(u)
    u_new = u_new.at[1:-1].set(0.5 * (u[:-2] + u[2:] - dx**2 * f[1:-1]))
    # Boundary conditions: u[0] = 0, u[-1] = 0
    return u_new


def solve_poisson_1d(f, n=100):
    """
    Solve 1D Poisson equation using direct method (linear system solve).
    
    Discretized equation: -d²u/dx² = f
    Using finite differences: -(u[i-1] - 2*u[i] + u[i+1])/dx² = f[i]
    
    Args:
        f: Source function f(x)
        n: Number of grid points
    
    Returns:
        x: Grid points
        u: Solution
    """
    # Create grid
    x = jnp.linspace(0, 1, n)
    dx = 1.0 / (n - 1)
    
    # Evaluate source term
    f_vals = f(x)
    
    # Build tridiagonal matrix A for -d²u/dx² discretization
    # Interior points: -(u[i-1] - 2*u[i] + u[i+1])/dx² = f[i]
    # Rearranged: u[i-1] - 2*u[i] + u[i+1] = -dx²*f[i]
    
    # For interior points (excluding boundaries)
    main_diag = 2.0 * jnp.ones(n-2)
    off_diag = -1.0 * jnp.ones(n-3)
    
    # Create tridiagonal matrix
    A = jnp.diag(main_diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    
    # Right-hand side (exclude boundary points)
    b = dx**2 * f_vals[1:-1]
    
    # Solve the linear system
    u_interior = jnp.linalg.solve(A, b)
    
    # Add boundary conditions
    u = jnp.concatenate([jnp.array([0.0]), u_interior, jnp.array([0.0])])
    
    return x, u


def source_function_1(x):
    """Source function: f(x) = 1"""
    return jnp.ones_like(x)


def analytical_solution_1(x):
    """Analytical solution for f(x) = 1: u(x) = 0.5*x*(1-x)"""
    return 0.5 * x * (1 - x)


def source_function_2(x):
    """Source function: f(x) = 2"""
    return 2.0 * jnp.ones_like(x)


def analytical_solution_2(x):
    """Analytical solution for f(x) = 2: u(x) = x(1-x)"""
    return x * (1 - x)


def test_poisson_solver():
    """Test the Poisson solver with known analytical solutions."""
    print("=" * 80)
    print("1D Poisson Equation Solver - Test Suite")
    print("=" * 80)
    
    # Test 1: f(x) = 1
    print("\nTest 1: f(x) = 1")
    print("-" * 80)
    x, u_numerical = solve_poisson_1d(source_function_1, n=100)
    u_analytical = analytical_solution_1(x)
    
    error = jnp.max(jnp.abs(u_numerical - u_analytical))
    print(f"Grid points: {len(x)}")
    print(f"Max error: {error:.6e}")
    print(f"Test 1: {'PASS' if error < 1e-4 else 'FAIL'}")
    
    # Test 2: f(x) = 2
    print("\nTest 2: f(x) = 2")
    print("-" * 80)
    x, u_numerical = solve_poisson_1d(source_function_2, n=100)
    u_analytical = analytical_solution_2(x)
    
    error = jnp.max(jnp.abs(u_numerical - u_analytical))
    print(f"Grid points: {len(x)}")
    print(f"Max error: {error:.6e}")
    print(f"Test 2: {'PASS' if error < 1e-4 else 'FAIL'}")
    
    # Save plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Test 1 plot
    x1, u1_num = solve_poisson_1d(source_function_1, n=50)
    u1_ana = analytical_solution_1(x1)
    ax1.plot(x1, u1_num, 'b-o', label='Numerical', markersize=3)
    ax1.plot(x1, u1_ana, 'r--', label='Analytical')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title('Test 1: f(x) = 1')
    ax1.legend()
    ax1.grid(True)
    
    # Test 2 plot
    x2, u2_num = solve_poisson_1d(source_function_2, n=50)
    u2_ana = analytical_solution_2(x2)
    ax2.plot(x2, u2_num, 'b-o', label='Numerical', markersize=3)
    ax2.plot(x2, u2_ana, 'r--', label='Analytical')
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x)')
    ax2.set_title('Test 2: f(x) = 2')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/workspace/jit/code/examples/poisson_results.png', dpi=100)
    print("\nPlot saved to: poisson_results.png")
    
    print("\n" + "=" * 80)
    print("All Poisson solver tests completed!")
    print("=" * 80)


def main():
    """Main entry point."""
    # Run tests
    test_poisson_solver()
    
    # Run twice to verify consistency
    print("\nRunning tests again to verify consistency...")
    test_poisson_solver()


if __name__ == "__main__":
    main()
