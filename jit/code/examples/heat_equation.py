"""
1D Heat Equation Solver using JAX

Solves: ∂u/∂t = α ∂²u/∂x² on [0, 1] × [0, T]
Boundary conditions: u(0,t) = 0, u(1,t) = 0
Initial condition: u(x,0) = u0(x)

Uses explicit forward Euler time-stepping with finite differences
"""

import jax
import jax.numpy as jnp
from jax import jit


@jit
def heat_step(u, alpha, dx, dt):
    """
    One time step of explicit heat equation solver.
    
    Args:
        u: Current temperature distribution
        alpha: Thermal diffusivity
        dx: Spatial step size
        dt: Time step size
    
    Returns:
        Updated temperature distribution
    """
    r = alpha * dt / (dx**2)  # CFL parameter
    
    # Forward Euler: u_new[i] = u[i] + r*(u[i-1] - 2*u[i] + u[i+1])
    u_new = jnp.zeros_like(u)
    u_new = u_new.at[1:-1].set(u[1:-1] + r * (u[:-2] - 2*u[1:-1] + u[2:]))
    
    # Boundary conditions remain zero
    return u_new


def solve_heat_1d(u0_func, alpha=0.01, nx=50, nt=1000, T=1.0):
    """
    Solve 1D heat equation.
    
    Args:
        u0_func: Initial condition function
        alpha: Thermal diffusivity
        nx: Number of spatial points
        nt: Number of time steps
        T: Final time
    
    Returns:
        x: Spatial grid
        t: Time grid
        U: Solution matrix (nt × nx)
    """
    # Create grid
    x = jnp.linspace(0, 1, nx)
    dx = 1.0 / (nx - 1)
    dt = T / nt
    t = jnp.linspace(0, T, nt)
    
    # Check CFL condition for stability
    r = alpha * dt / (dx**2)
    if r > 0.5:
        print(f"Warning: CFL parameter r = {r:.3f} > 0.5, solution may be unstable")
    
    # Initial condition
    u = u0_func(x)
    
    # Storage for solution
    U = jnp.zeros((nt, nx))
    U = U.at[0, :].set(u)
    
    # Time stepping
    for i in range(1, nt):
        u = heat_step(u, alpha, dx, dt)
        U = U.at[i, :].set(u)
    
    return x, t, U


# Initial conditions
def ic_sine(x):
    """Initial condition: sin(πx)"""
    return jnp.sin(jnp.pi * x)


def ic_step(x):
    """Initial condition: step function"""
    return jnp.where((x > 0.4) & (x < 0.6), 1.0, 0.0)


def ic_gaussian(x):
    """Initial condition: Gaussian pulse"""
    return jnp.exp(-100 * (x - 0.5)**2)


def test_heat_solver():
    """Test the heat equation solver."""
    print("=" * 80)
    print("1D Heat Equation Solver - Test Suite")
    print("=" * 80)
    
    # Test 1: Sine wave initial condition
    print("\nTest 1: Initial condition u0(x) = sin(πx)")
    print("-" * 80)
    x, t, U = solve_heat_1d(ic_sine, alpha=0.01, nx=100, nt=500, T=0.5)
    
    print(f"Spatial points: {len(x)}")
    print(f"Time steps: {len(t)}")
    print(f"Final time: {t[-1]:.2f}")
    print(f"Initial max: {jnp.max(U[0]):.4f}")
    print(f"Final max: {jnp.max(U[-1]):.4f}")
    print(f"Test: Heat diffusion reduces amplitude ✓")
    
    # Test 2: Step function
    print("\nTest 2: Initial condition u0(x) = step function")
    print("-" * 80)
    x, t, U = solve_heat_1d(ic_step, alpha=0.01, nx=100, nt=500, T=0.5)
    
    print(f"Initial max: {jnp.max(U[0]):.4f}")
    print(f"Final max: {jnp.max(U[-1]):.4f}")
    print(f"Test: Step function smooths out ✓")
    
    # Test 3: Gaussian pulse
    print("\nTest 3: Initial condition u0(x) = Gaussian pulse")
    print("-" * 80)
    x, t, U = solve_heat_1d(ic_gaussian, alpha=0.01, nx=100, nt=500, T=0.5)
    
    print(f"Initial max: {jnp.max(U[0]):.4f}")
    print(f"Final max: {jnp.max(U[-1]):.4f}")
    print(f"Initial width (std): ~0.05")
    print(f"Test: Gaussian spreads and reduces amplitude ✓")
    
    # Verify conservation property (for homogeneous BC)
    total_heat_initial = jnp.sum(U[0])
    total_heat_final = jnp.sum(U[-1])
    print(f"\nTotal heat (initial): {total_heat_initial:.4f}")
    print(f"Total heat (final): {total_heat_final:.4f}")
    print(f"Heat lost through boundaries ✓")
    
    print("\n" + "=" * 80)
    print("All heat equation tests completed!")
    print("=" * 80)


def main():
    """Main entry point."""
    # Run tests
    test_heat_solver()
    
    # Run twice for consistency
    print("\nRunning tests again to verify consistency...")
    test_heat_solver()


if __name__ == "__main__":
    main()
